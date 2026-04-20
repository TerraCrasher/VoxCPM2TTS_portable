"""VoxCPM2 Voice Cloner - Gradio GUI"""

import os
import gc
import time
import threading
from pathlib import Path
from datetime import datetime
from app.config import VERSION

import gradio as gr
import soundfile as sf

from app.config import (
    BASE_DIR, OUTPUT_DIR, INPUT_DIR, LORA_DIR, JSONL_DIR,
    DEFAULT_CFG_VALUE, DEFAULT_INFERENCE_TIMESTEPS,
    STYLE_PRESETS,
)

# ── 전역 ──
cloner = None
is_generating = False
cancel_requested = False
result_file_map = {}

MODE_DESC = {
    "Controllable": "🎛️ 참조 음성의 목소리 + 스타일 제어 가능 (정밀도 ⭐⭐⭐)",
    "Ultimate": "🎙️ 참조 음성의 모든 뉘앙스 정밀 복제 (정밀도 ⭐⭐⭐⭐⭐)",
    "Two-pass": "🔁 Ultimate 정밀도 + 스타일 제어 (정밀도 ⭐⭐⭐⭐, 시간 2배)",
    "LoRA 일반": "🎓 학습된 화자 음색으로 생성 (정밀도 ⭐⭐⭐⭐⭐)",
    "LoRA + 감정 제어": "🎓 학습된 화자 음색 + 스타일 제어 (정밀도 ⭐⭐⭐⭐)",
    "Voice Design": "🎨 참조 음성 없이 설명만으로 새 목소리 생성",
}


def get_cloner():
    global cloner
    if cloner is None:
        from app.engine import VoiceCloner
        cloner = VoiceCloner()
    return cloner


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────
def find_reference_audio():
    extensions = {".wav", ".mp3", ".flac", ".ogg"}
    files = []
    for f in INPUT_DIR.rglob("*"):
        if f.suffix.lower() in extensions:
            files.append(str(f))
    return sorted(files)


def ref_display_name(path):
    p = Path(path)
    try:
        rel = p.relative_to(INPUT_DIR)
        return str(rel)
    except ValueError:
        return p.name


def find_lora_checkpoints():
    if not LORA_DIR.exists():
        return []
    checkpoints = []
    for model_dir in sorted(LORA_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        ckpt_dir = model_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        for step_dir in sorted(ckpt_dir.iterdir()):
            if step_dir.is_dir() and (step_dir / "lora_weights.safetensors").exists():
                checkpoints.append(str(step_dir))
    return checkpoints

def find_jsonl_files():
    files = sorted(JSONL_DIR.glob("*.jsonl"))
    return [f.name for f in files]


def format_lora_name(path):
    p = Path(path)
    return f"{p.parent.parent.name}/{p.name}"


def get_style_choices():
    choices = ["없음"]
    for key, val in STYLE_PRESETS.items():
        if key == "0":
            continue
        choices.append(f"{val['name']} → {val['prompt']}")
    choices.append("직접 입력")
    return choices


def style_to_prompt(style_choice, custom_style):
    parts = []
    # 프리셋
    if style_choice and style_choice != "없음" and style_choice != "직접 입력":
        if "→" in style_choice:
            parts.append(style_choice.split("→")[1].strip())
    # 직접 입력
    if custom_style and custom_style.strip():
        parts.append(custom_style.strip())
    return ", ".join(parts)


def open_folder_safe():
    try:
        path = str(OUTPUT_DIR)
        threading.Thread(target=lambda: os.system(f'explorer "{path}"'), daemon=True).start()
        return "📂 폴더 열림"
    except Exception as e:
        return f"❌ 폴더 열기 실패: {e}"


def request_cancel():
    global cancel_requested
    cancel_requested = True
    return "⛔ 취소 요청됨... 즉시 중단합니다."


# ─────────────────────────────────────────────
# 음성 생성
# ─────────────────────────────────────────────
def generate_voice(
    mode, reference_audio, uploaded_audio, text,
    cfg_value, inference_timesteps, count,
    style_choice, custom_style,
    lora_path, voice_desc,
):
    global is_generating, cancel_requested, result_file_map

    if is_generating:
        yield None, gr.update(), "⚠️ 이미 생성 중입니다."
        return

    is_generating = True
    cancel_requested = False
    total_start = time.time()

    try:
        c = get_cloner()

        # ── 입력 검증 ──
        if not text or not text.strip():
            yield None, gr.update(), "❌ 텍스트를 입력하세요."
            return

        # Voice Design은 참조 음성 불필요
        ref_audio = None
        if mode != "Voice Design":
            ref_audio = uploaded_audio if uploaded_audio else reference_audio
            if not ref_audio:
                yield None, gr.update(), "❌ 참조 음성을 선택하거나 업로드하세요."
                return
            if not Path(ref_audio).exists():
                yield None, gr.update(), "❌ 참조 음성 파일이 존재하지 않습니다."
                return

        if mode == "Voice Design":
            if not voice_desc or not voice_desc.strip():
                yield None, gr.update(), "❌ 음성 설명을 입력하세요.\n  예: A young woman, gentle and sweet voice"
                return

        # LoRA 검증
        target_lora = None
        if mode in ["LoRA 일반", "LoRA + 감정 제어"]:
            if not lora_path or lora_path == "없음" or lora_path == "":
                yield None, gr.update(), "❌ LoRA 체크포인트를 선택하세요."
                return
            if not Path(lora_path).exists():
                yield None, gr.update(), "❌ LoRA 체크포인트가 존재하지 않습니다."
                return
            target_lora = lora_path

        # 파라미터 검증
        try:
            cfg_value = float(cfg_value)
            inference_timesteps = int(inference_timesteps)
            count = max(1, int(count))
        except (ValueError, TypeError):
            yield None, gr.update(), "❌ 파라미터 값이 올바르지 않습니다."
            return

        # ── 모델 로드 ──
        if not c.is_loaded or c.current_lora != target_lora:
            yield None, gr.update(), "🔄 모델 로딩 중... (~20초)"
            try:
                c.load_model(lora_path=target_lora)
            except Exception as e:
                yield None, gr.update(), f"❌ 모델 로드 실패:\n{e}"
                return

        # ── 스타일 처리 ──
        style = style_to_prompt(style_choice, custom_style)
        if style:
            styled_text = f"({style}){text}"
        else:
            styled_text = text

        # ── Ultimate 대본 추출 (1회만) ──
        prompt_text = None
        if mode in ["Ultimate", "Two-pass"]:
            yield None, gr.update(), "📝 대본 자동 추출 중 (Whisper)..."
            prompt_text = c.transcribe(ref_audio)

        # ── 생성 루프 ──
        results = []
        for i in range(count):
            # 취소 확인
            if cancel_requested:
                elapsed = time.time() - total_start
                if results:
                    result_file_map.clear()
                    result_file_map.update({Path(f).name: f for f in results})
                    display_names = list(result_file_map.keys())
                    file_list = "\n".join([f"  ✅ {n}" for n in display_names])
                    yield results[0], gr.update(choices=display_names, value=display_names[0], visible=True), (
                        f"⛔ 취소됨! ({len(results)}/{count}개 완료)\n"
                        f"  ⏱️ 소요: {elapsed:.1f}초\n\n{file_list}\n\n📁 {OUTPUT_DIR}"
                    )
                else:
                    yield None, gr.update(visible=False), f"⛔ 취소됨!\n  ⏱️ 소요: {elapsed:.1f}초"
                return

            # 진행 표시
            bar_len = 20
            filled = int((i / count) * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            pct = int(i / count * 100)
            elapsed = time.time() - total_start
            yield None, gr.update(), (
                f"🔊 생성 중... [{i+1}/{count}]\n"
                f"  전체: [{bar}] {pct}%\n\n"
                f"  완료: {len(results)}개 | 남음: {count - i}개\n"
                f"  ⏱️ 경과: {elapsed:.0f}초"
            )

            try:
                # ── 모드별 생성 ──
                if mode == "Voice Design":
                    gen_kwargs = {
                        "text": f"({voice_desc}){text}",
                        "cfg_value": cfg_value,
                        "inference_timesteps": inference_timesteps,
                    }
                    wav, gen_time = c._generate_once(
                        gen_kwargs, cancel_check=lambda: cancel_requested,
                    )

                elif mode == "Two-pass":
                    # Pass 1: Ultimate
                    yield None, gr.update(), (
                        f"🔁 Two-pass [{i+1}/{count}] - Pass 1/2\n"
                        f"  Ultimate 정밀 복제 중...\n"
                        f"  ⏱️ 경과: {time.time() - total_start:.0f}초"
                    )
                    pass1_kwargs = {
                        "text": text,
                        "reference_wav_path": ref_audio,
                        "cfg_value": cfg_value,
                        "inference_timesteps": inference_timesteps,
                        "prompt_wav_path": ref_audio,
                        "prompt_text": prompt_text,
                    }
                    wav1, _ = c._generate_once(
                        pass1_kwargs, cancel_check=lambda: cancel_requested,
                    )

                    if style:
                        # 임시 저장
                        temp_name = f"temp_twopass_{datetime.now().strftime('%H%M%S')}.wav"
                        temp_path = OUTPUT_DIR / temp_name
                        sf.write(str(temp_path), wav1, c.model.tts_model.sample_rate)

                        yield None, gr.update(), (
                            f"🔁 Two-pass [{i+1}/{count}] - Pass 2/2\n"
                            f"  스타일 적용 중: {style}\n"
                            f"  ⏱️ 경과: {time.time() - total_start:.0f}초"
                        )
                        pass2_kwargs = {
                            "text": styled_text,
                            "reference_wav_path": str(temp_path),
                            "cfg_value": cfg_value,
                            "inference_timesteps": inference_timesteps,
                        }
                        wav, gen_time = c._generate_once(
                            pass2_kwargs, cancel_check=lambda: cancel_requested,
                        )
                        try:
                            os.remove(str(temp_path))
                        except OSError:
                            pass
                    else:
                        wav, gen_time = wav1, 0

                elif mode == "Ultimate":
                    gen_kwargs = {
                        "text": text,
                        "reference_wav_path": ref_audio,
                        "cfg_value": cfg_value,
                        "inference_timesteps": inference_timesteps,
                        "prompt_wav_path": ref_audio,
                        "prompt_text": prompt_text,
                    }
                    wav, gen_time = c._generate_once(
                        gen_kwargs, cancel_check=lambda: cancel_requested,
                    )

                elif mode in ["Controllable", "LoRA + 감정 제어"]:
                    gen_kwargs = {
                        "text": styled_text,
                        "reference_wav_path": ref_audio,
                        "cfg_value": cfg_value,
                        "inference_timesteps": inference_timesteps,
                    }
                    wav, gen_time = c._generate_once(
                        gen_kwargs, cancel_check=lambda: cancel_requested,
                    )

                elif mode == "LoRA 일반":
                    gen_kwargs = {
                        "text": text,
                        "reference_wav_path": ref_audio,
                        "cfg_value": cfg_value,
                        "inference_timesteps": inference_timesteps,
                    }
                    wav, gen_time = c._generate_once(
                        gen_kwargs, cancel_check=lambda: cancel_requested,
                    )

                else:
                    yield None, gr.update(), "❌ 알 수 없는 모드"
                    return

                # 파일 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = "design" if mode == "Voice Design" else "clone"
                if count == 1:
                    fname = f"{prefix}_{timestamp}.wav"
                else:
                    fname = f"{prefix}_{timestamp}_{i+1:02d}.wav"

                output_path = OUTPUT_DIR / fname
                sf.write(str(output_path), wav, c.model.tts_model.sample_rate)
                results.append(str(output_path))

                elapsed = time.time() - total_start
                done_pct = int(((i + 1) / count) * 100)
                done_bar = "█" * ((i + 1) * 20 // count) + "░" * (20 - (i + 1) * 20 // count)
                yield None, gr.update(), (
                    f"🔊 [{i+1}/{count}] 완료 ✅\n"
                    f"  전체: [{done_bar}] {done_pct}%\n"
                    f"  저장: {fname}\n"
                    f"  ⏱️ 경과: {elapsed:.0f}초"
                )

            except InterruptedError:
                elapsed = time.time() - total_start
                if results:
                    result_file_map.clear()
                    result_file_map.update({Path(f).name: f for f in results})
                    display_names = list(result_file_map.keys())
                    file_list = "\n".join([f"  ✅ {n}" for n in display_names])
                    yield results[0], gr.update(choices=display_names, value=display_names[0], visible=True), (
                        f"⛔ 즉시 취소됨! ({len(results)}/{count}개 완료)\n"
                        f"  ⏱️ 소요: {elapsed:.1f}초\n\n{file_list}\n\n📁 {OUTPUT_DIR}"
                    )
                else:
                    yield None, gr.update(visible=False), f"⛔ 즉시 취소됨!\n  ⏱️ 소요: {elapsed:.1f}초"
                return

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    import torch
                    torch.cuda.empty_cache()
                    gc.collect()
                    yield None, gr.update(), (
                        "❌ VRAM 부족!\n\n"
                        "  1. 다른 프로그램 종료\n"
                        "  2. CFG 낮추기\n"
                        "  3. Steps 줄이기\n"
                        "  4. 생성 개수 줄이기"
                    )
                    return
                else:
                    yield None, gr.update(), f"❌ 생성 에러:\n{e}"
                    return

        # ── 완료 ──
        total_elapsed = time.time() - total_start
        if results:
            result_file_map.clear()
            result_file_map.update({Path(f).name: f for f in results})
            display_names = list(result_file_map.keys())
            file_list = "\n".join([f"  ✅ {n}" for n in display_names])
            yield results[0], gr.update(choices=display_names, value=display_names[0], visible=True), (
                f"🔔 생성 완료! ({len(results)}개)\n"
                f"  [████████████████████] 100%\n"
                f"  ⏱️ 총 소요: {total_elapsed:.1f}초\n\n"
                f"{file_list}\n\n📁 {OUTPUT_DIR}"
            )
        else:
            yield None, gr.update(visible=False), "❌ 생성 결과가 없습니다."

    except Exception as e:
        yield None, gr.update(), f"❌ 예기치 않은 에러:\n{e}"

    finally:
        is_generating = False


# ─────────────────────────────────────────────
# LoRA 학습
# ─────────────────────────────────────────────
def start_training(
    output_name, manifest_path,
    learning_rate, max_iterations, save_interval,
    lora_rank, lora_alpha,
):
    global is_generating
    if is_generating:
        return "⚠️ 음성 생성 중입니다."

    from app.trainer import create_config, run_training, find_model_path

    if not output_name:
        return "❌ 학습 이름을 입력하세요."
    if not manifest_path or not (JSONL_DIR / manifest_path).exists():
        return f"❌ 파일 없음: {JSONL_DIR / manifest_path}"

    model_path = find_model_path()
    if not model_path:
        return "❌ VoxCPM2 모델을 찾을 수 없습니다."

    try:
        config_path = create_config(
            output_name=output_name, 
            train_manifest=str(JSONL_DIR / manifest_path),  # 전체 경로
            model_path=model_path, learning_rate=float(learning_rate),
            max_iterations=int(max_iterations), batch_size=1,
            lora_rank=int(lora_rank), lora_alpha=int(lora_alpha),
            save_interval=int(save_interval),
        )
        result = run_training(config_path)
        if result == 0:
            return f"✅ 학습 완료!\n   저장: {LORA_DIR / output_name}"
        else:
            return f"⚠️ 학습 종료 (코드: {result})\n   체크포인트는 저장됨"
    except Exception as e:
        return f"❌ 학습 에러: {e}"


# ─────────────────────────────────────────────
# UI 구성
# ─────────────────────────────────────────────
def create_ui():
    ref_audio_list = find_reference_audio()
    ref_display_list = [ref_display_name(p) for p in ref_audio_list]

    lora_list = find_lora_checkpoints()
    lora_choices = ["없음"] + [format_lora_name(p) for p in lora_list]

    with gr.Blocks(title="VoxCPM2 Voice Cloner", theme=gr.themes.Soft()) as app:

        # 업데이트 체크
        update_msg = ""
        try:
            from app.updater import check_update
            update_info = check_update()
            if update_info["available"]:
                update_msg = (
                    f"\n> 🔔 **새 버전 v{update_info['latest']}** 이 있습니다! "
                    f"[다운로드]({update_info['url']}/releases/latest) | "
                    f"`update.bat` 실행으로 업데이트"
                )
        except Exception:
            pass

        gr.Markdown(f"# 🎙️ VoxCPM2 Voice Cloner v{VERSION}")
        if update_msg:
            gr.Markdown(update_msg)

        with gr.Tabs():
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 탭 1: 음성 생성
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("🔊 음성 생성"):
                with gr.Row():
                    with gr.Column(scale=3):

                        # STEP 1
                        gr.Markdown("### STEP 1 — 모드 선택")
                        mode = gr.Radio(
                            choices=["Controllable", "Ultimate", "Two-pass",
                                     "LoRA 일반", "LoRA + 감정 제어", "Voice Design"],
                            value="Controllable", label="클로닝 모드",
                        )
                        mode_desc = gr.Markdown(f"_{MODE_DESC['Controllable']}_")

                        # STEP 2
                        gr.Markdown("### STEP 2 — 입력")

                        with gr.Row():
                            reference_audio = gr.Dropdown(
                                choices=ref_display_list,
                                label="참조 음성 (목록에서 선택)",
                                interactive=True,
                            )
                            ref_refresh = gr.Button("🔄", scale=0, min_width=50)

                        uploaded_audio = gr.Audio(
                            label="또는 파일 드래그 앤 드롭",
                            type="filepath", sources=["upload"],
                        )

                        ref_preview = gr.Audio(label="참조 음성 미리듣기", type="filepath")

                        voice_desc = gr.Textbox(
                            label="🎨 음성 설명 (Voice Design용, 영어 권장)",
                            placeholder="예: A young woman, gentle and sweet voice\n예: mature male, deep and calm voice",
                            lines=2, visible=False,
                        )

                        text = gr.Textbox(
                            label="생성할 텍스트",
                            placeholder="여기에 텍스트를 입력하세요...", lines=3,
                        )

                        # STEP 3
                        gr.Markdown("### STEP 3 — 옵션")

                        lora_select = gr.Dropdown(
                            choices=lora_choices, value="없음",
                            label="LoRA 체크포인트", interactive=True, visible=False,
                        )
                        lora_refresh = gr.Button("🔄 LoRA 새로고침", visible=False)

                        with gr.Row():
                            style_choice = gr.Dropdown(
                                choices=get_style_choices(), value="없음",
                                label="스타일 프리셋", interactive=True,
                            )
                            custom_style = gr.Textbox(
                                label="직접 입력 (영어)",
                                placeholder="cheerful, slightly faster",
                            )
                        style_hint = gr.Markdown("_💡 프리셋 + 직접 입력 조합 가능 (예: 프리셋 '빠르게' + 직접 입력 'angry' → 'slightly faster, angry')_")

                        with gr.Row():
                            cfg_value = gr.Slider(
                                minimum=0, maximum=10, step=0.5,
                                value=DEFAULT_CFG_VALUE, label="CFG",
                            )
                            inference_timesteps = gr.Slider(
                                minimum=1, maximum=30, step=1,
                                value=DEFAULT_INFERENCE_TIMESTEPS, label="Steps",
                            )
                            count = gr.Number(
                                value=1, label="생성 개수", minimum=1, precision=0,
                            )

                        with gr.Row():
                            generate_btn = gr.Button("🔊 음성 생성", variant="primary", size="lg")
                            cancel_btn = gr.Button("⛔ 취소", variant="stop", size="lg", visible=False)
                            reset_btn = gr.Button("🔄 초기화", size="lg")

                    # ── 오른쪽: 결과 ──
                    with gr.Column(scale=2):
                        gr.Markdown("### 결과")
                        result_selector = gr.Dropdown(
                            choices=[], label="생성된 파일 선택",
                            interactive=True, visible=False,
                        )
                        output_audio = gr.Audio(label="생성된 음성", type="filepath")
                        status_text = gr.Textbox(label="상태", lines=8, interactive=False)
                        open_folder_btn = gr.Button("📂 출력 폴더 열기")

                # ── State ──
                lora_path_state = gr.State("")
                ref_path_state = gr.State("")

                # ── 이벤트: 모드 변경 ──
                def update_mode_desc(m):
                    desc = MODE_DESC.get(m, "")
                    show_lora = m in ["LoRA 일반", "LoRA + 감정 제어"]
                    show_style = m in ["Controllable", "Two-pass", "LoRA + 감정 제어"]
                    show_ref = m != "Voice Design"
                    show_voice_desc = m == "Voice Design"
                    return (
                        f"_{desc}_",
                        gr.update(visible=show_lora),
                        gr.update(visible=show_lora),
                        gr.update(visible=show_style),
                        gr.update(visible=show_style),
                        gr.update(visible=show_ref),
                        gr.update(visible=show_ref),
                        gr.update(visible=show_ref),
                        gr.update(visible=show_ref),
                        gr.update(visible=show_voice_desc),
                        gr.update(visible=show_style),  # style_hint
                    )

                mode.change(
                    fn=update_mode_desc, inputs=mode,
                    outputs=[
                        mode_desc, lora_select, lora_refresh,
                        style_choice, custom_style,
                        reference_audio, ref_refresh, uploaded_audio, ref_preview,
                        voice_desc, style_hint,
                    ],
                )

                # ── 이벤트: 참조 음성 ──
                def refresh_refs():
                    new_list = find_reference_audio()
                    return gr.update(choices=[ref_display_name(p) for p in new_list])

                ref_refresh.click(fn=refresh_refs, outputs=reference_audio)

                def on_ref_select(display_name):
                    for p in find_reference_audio():
                        if ref_display_name(p) == display_name:
                            return p, p, gr.update(value=None)
                    return "", None, gr.update()

                reference_audio.change(
                    fn=on_ref_select, inputs=reference_audio,
                    outputs=[ref_path_state, ref_preview, uploaded_audio],
                )

                def on_upload(filepath):
                    if filepath and Path(filepath).exists():
                        return filepath, gr.update(value=None)
                    return None, gr.update()

                uploaded_audio.change(
                    fn=on_upload, inputs=uploaded_audio,
                    outputs=[ref_preview, reference_audio],
                )

                # ── 이벤트: LoRA ──
                def refresh_loras():
                    new_list = find_lora_checkpoints()
                    return gr.update(choices=["없음"] + [format_lora_name(p) for p in new_list])

                lora_refresh.click(fn=refresh_loras, outputs=lora_select)

                def resolve_lora(display_name):
                    if display_name == "없음":
                        return ""
                    for p in find_lora_checkpoints():
                        if format_lora_name(p) == display_name:
                            return p
                    return ""

                lora_select.change(fn=resolve_lora, inputs=lora_select, outputs=lora_path_state)

                # ── 이벤트: 결과 파일 선택 ──
                def on_result_select(display_name):
                    path = result_file_map.get(display_name, "")
                    if path and Path(path).exists():
                        return path
                    return None

                result_selector.change(fn=on_result_select, inputs=result_selector, outputs=output_audio)

                # ── 이벤트: 초기화 ──
                def reset_all():
                    return (
                        gr.update(value="Controllable"),
                        gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=""),
                        gr.update(value="없음"),
                        gr.update(value=""),
                        gr.update(value=DEFAULT_CFG_VALUE),
                        gr.update(value=DEFAULT_INFERENCE_TIMESTEPS),
                        gr.update(value=1),
                        gr.update(value=None),
                        gr.update(visible=False),
                        gr.update(value="🔄 초기화 완료"),
                        gr.update(value=""),
                    )

                reset_btn.click(
                    fn=reset_all,
                    outputs=[
                        mode, reference_audio, uploaded_audio, ref_preview,
                        text, style_choice, custom_style,
                        cfg_value, inference_timesteps, count,
                        output_audio, result_selector, status_text, voice_desc,
                    ],
                )

                # ── 이벤트: 취소 ──
                cancel_btn.click(fn=request_cancel, outputs=status_text)

                # ── 이벤트: 생성 ──
                generate_btn.click(
                    fn=lambda c: (
                        gr.update(interactive=False),
                        gr.update(visible=int(c) > 1),
                        "⏳ 생성 준비 중...",
                    ),
                    inputs=count,
                    outputs=[generate_btn, cancel_btn, status_text],
                ).then(
                    fn=generate_voice,
                    inputs=[
                        mode, ref_path_state, uploaded_audio, text,
                        cfg_value, inference_timesteps, count,
                        style_choice, custom_style, lora_path_state, voice_desc,
                    ],
                    outputs=[output_audio, result_selector, status_text],
                    show_progress="hidden",
                ).then(
                    fn=lambda: (gr.update(interactive=True), gr.update(visible=False)),
                    outputs=[generate_btn, cancel_btn],
                )

                # ── 이벤트: 폴더 열기 ──
                open_folder_btn.click(fn=open_folder_safe, outputs=status_text)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 탭 2: LoRA 학습
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("🎓 LoRA 학습"):
                gr.Markdown("### LoRA Fine-tuning")
                gr.Markdown("_5~10분 분량의 음성 데이터로 특정 화자를 학습합니다._")

                with gr.Tabs():
                    # ── 학습 데이터 준비 ──
                    with gr.Tab("📝 1. 학습 데이터 준비"):
                        gr.Markdown("""
### 학습 데이터 준비 (Whisper 자동 대본 추출)

**사용법:**
1. `inputs/training_data/` 폴더에 음성 파일(wav)을 넣으세요
2. 음성은 **5~15초 단위 클립**으로 분할된 상태여야 합니다
3. 아래 버튼을 누르면 자동으로 대본을 추출하여 `jsonl`을 생성합니다
""")
                        with gr.Row():
                            prep_language = gr.Dropdown(
                                choices=["ja", "ko", "en", "zh"],
                                value="ja",
                                label="음성 언어",
                            )
                            prep_min_duration = gr.Number(
                                value=0.5,
                                label="최소 길이 (초)",
                            )
                            prep_max_duration = gr.Number(
                                value=30.0,
                                label="최대 길이 (초)",
                            )

                        prep_output_name = gr.Textbox(
                            label="출력 파일명",
                            value="training_manifest.jsonl",
                            placeholder="예: ashe_training.jsonl",
                        )

                        prep_btn = gr.Button("📝 학습 데이터 생성", variant="primary", size="lg")
                        prep_status = gr.Textbox(
                            label="준비 상태",
                            lines=10,
                            interactive=False,
                        )


                        def prepare_training_data(language, min_dur, max_dur, output_name):
                            import json
                            import warnings

                            training_dir = BASE_DIR / "inputs" / "training_data"

                            extensions = {".wav", ".mp3", ".flac", ".ogg"}
                            audio_files = sorted([
                                f for f in training_dir.iterdir()
                                if f.suffix.lower() in extensions
                            ]) if training_dir.exists() else []

                            if not audio_files:
                                yield f"❌ 음성 파일이 없습니다.\n   📁 여기에 넣으세요: {training_dir}"
                                return

                            yield f"📂 음성 파일 {len(audio_files)}개 발견\n🔄 Whisper 모델 로딩 중..."

                            try:
                                import whisper
                                import soundfile as sf_lib
                            except ImportError:
                                yield "❌ whisper 또는 soundfile이 설치되어 있지 않습니다."
                                return

                            warnings.filterwarnings("ignore", message=".*Performing inference on CPU.*")
                            warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
                            model = whisper.load_model("base", device="cpu")

                            yield f"📂 음성 파일 {len(audio_files)}개 발견\n✅ Whisper 로드 완료\n\n📝 대본 추출 시작..."

                            entries = []
                            errors = []
                            total = len(audio_files)

                            for idx, audio_file in enumerate(audio_files):
                                pct = int((idx / total) * 100)
                                filled = idx * 20 // total
                                bar = "█" * filled + "░" * (20 - filled)

                                try:
                                    info = sf_lib.info(str(audio_file))
                                    duration = round(info.duration, 2)

                                    if duration < float(min_dur):
                                        status = f"⚠️ 스킵 (짧음 {duration}초)"
                                    elif duration > float(max_dur):
                                        status = f"⚠️ 스킵 (김 {duration}초)"
                                    else:
                                        result_obj = model.transcribe(str(audio_file), language=language)
                                        text = result_obj["text"].strip()

                                        if not text:
                                            status = "⚠️ 스킵 (텍스트 없음)"
                                        else:
                                            entries.append({
                                                "audio": str(audio_file.resolve()),
                                                "text": text,
                                                "duration": duration,
                                            })
                                            status = f"✅ {text[:40]}"

                                except Exception as e:
                                    errors.append((audio_file.name, str(e)))
                                    status = f"❌ {e}"

                                yield (
                                    f"📝 대본 추출 중... [{idx+1}/{total}]\n"
                                    f"  [{bar}] {pct}%\n\n"
                                    f"  현재: {audio_file.name}\n"
                                    f"  {status}\n\n"
                                    f"  완료: {len(entries)}개 | 에러: {len(errors)}개"
                                )

                            # 저장
                            if entries:
                                output_jsonl = JSONL_DIR / output_name
                                with open(output_jsonl, "w", encoding="utf-8") as f:
                                    for entry in entries:
                                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                                total_dur = sum(e["duration"] for e in entries)
                                file_list = "\n".join([f"  · {Path(e['audio']).name}" for e in entries[:10]])
                                if len(entries) > 10:
                                    file_list += f"\n  ... 외 {len(entries)-10}개"

                                yield (
                                    f"🎉 완료!\n"
                                    f"  [████████████████████] 100%\n\n"
                                    f"  📄 저장: {output_jsonl.name}\n"
                                    f"  📊 총 {len(entries)}개 클립\n"
                                    f"  ⏱️ 총 길이: {total_dur:.1f}초 ({total_dur/60:.1f}분)\n\n"
                                    f"{file_list}"
                                )
                            else:
                                yield "❌ 추출된 데이터가 없습니다."

                            # Whisper 해제
                            del model
                            gc.collect()

                            return "\n".join(result_lines)

                        prep_btn.click(
                            fn=prepare_training_data,
                            inputs=[prep_language, prep_min_duration, prep_max_duration, prep_output_name],
                            outputs=prep_status,
                        )

                    # ── 학습 실행 ──
                    with gr.Tab("🎓 2. 학습 실행"):
                        with gr.Row():
                            with gr.Column():
                                train_name = gr.Textbox(label="학습 이름", placeholder="예: ashe_lora")
                                with gr.Row():
                                    train_manifest = gr.Dropdown(
                                        choices=find_jsonl_files(),
                                        label="Train Manifest",
                                        interactive=True,
                                        allow_custom_value=True,
                                    )
                                    manifest_refresh = gr.Button("🔄", scale=0, min_width=50)
                            with gr.Column():
                                train_lr = gr.Number(label="Learning Rate", value=0.0001)
                                train_iters = gr.Number(label="Max Iterations", value=5000)
                                train_save = gr.Number(label="Save Interval", value=500)

                        with gr.Row():
                            train_rank = gr.Slider(minimum=4, maximum=128, step=4, value=32, label="LoRA Rank")
                            train_alpha = gr.Slider(minimum=4, maximum=128, step=4, value=16, label="LoRA Alpha")

                        train_btn = gr.Button("🎓 학습 시작", variant="primary", size="lg")
                        train_status = gr.Textbox(
                            label="학습 상태 (터미널에서 상세 로그 확인)",
                            lines=10, interactive=False,
                            placeholder="학습 시작 버튼을 누르세요.\n진행 로그는 터미널(CMD)에서 실시간 확인 가능합니다.",
                        )

                        def refresh_manifests():
                            return gr.update(choices=find_jsonl_files())

                        manifest_refresh.click(fn=refresh_manifests, outputs=train_manifest)

                        train_btn.click(
                            fn=start_training,
                            inputs=[train_name, train_manifest, train_lr, train_iters, train_save, train_rank, train_alpha],
                            outputs=train_status,
                        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7870, share=False, inbrowser=True)