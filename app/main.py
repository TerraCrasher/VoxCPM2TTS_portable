"""VoxCPM2 음성 클로닝 - CLI 메인 실행"""

from pathlib import Path
from app.engine import VoiceCloner
from app.config import (
    DEFAULT_CFG_VALUE, DEFAULT_INFERENCE_TIMESTEPS,
    STYLE_PRESETS, BASE_DIR, LORA_DIR,
)
from app.utils import print_audio_list, setup_logger, open_output_folder, play_audio


def main():
    logger = setup_logger()
    cloner = VoiceCloner()

    print("\n" + "=" * 50)
    print("  🎙️ VoxCPM2 음성 클로닝 프로그램")
    print("=" * 50)

    # 업데이트 체크
    try:
        from app.updater import check_update
        update_info = check_update()
        print(f"\n  버전: v{update_info['current']}")
        if update_info["available"]:
            print(f"  {update_info['message']}")
    except Exception:
        pass

    cloner.load_model()

    while True:
        print("\n" + "=" * 50)
        print("  📌 메인 메뉴")
        print("=" * 50)
        print("  [1] 음성 클로닝")
        print("  [2] LoRA 적용 클로닝")
        print("  [3] LoRA 학습")
        print("  [4] Voice Design (참조 음성 없이)")
        print("  [5] 참조 음성 목록")
        print("  [6] 출력 폴더 열기")
        print("  [q] 종료")

        choice = input("\n선택 [기본=1]: ").strip()
        if not choice:
            choice = "1"

        if choice == "q":
            print("\n👋 프로그램을 종료합니다.")
            break
        elif choice == "1":
            menu_clone(cloner, logger)
        elif choice == "2":
            menu_lora_clone(cloner, logger)
        elif choice == "3":
            menu_lora_train()
        elif choice == "4":
            try:
                result = run_voice_design(cloner)
                if result:
                    logger.info(f"Voice Design 완료: {result}")
                    after_generate(result)
            except KeyboardInterrupt:
                print("\n⚠️ 취소됨")
            except Exception as e:
                print(f"\n❌ 에러: {e}")
                logger.error(f"에러: {e}")
        elif choice == "5":
            print_audio_list()
        elif choice == "6":
            open_output_folder()
        else:
            print("⚠️ 올바른 메뉴를 선택하세요.")


# ─────────────────────────────────────────────
# 1. 음성 클로닝
# ─────────────────────────────────────────────
def menu_clone(cloner, logger):
    if cloner.current_lora:
        print("\n⚠️ LoRA 적용 상태 → 기본 모델로 전환합니다. (~20초 소요)")
        confirm = input("  계속? (y/n) [y]: ").strip()
        if confirm.lower() == "n":
            return
        cloner.load_model(lora_path=None)

    print("\n  📌 음성 클로닝 모드:")
    print("  [1] Controllable (+ 스타일 제어)")
    print("  [2] Ultimate (정밀 복제)")
    print("  [3] Two-pass (정밀도 + 스타일)")
    print("  [b] 돌아가기")

    sub = input("\n  선택 [기본=1]: ").strip()
    if not sub:
        sub = "1"
    if sub == "b":
        return

    try:
        if sub == "1":
            result = run_controllable(cloner)
        elif sub == "2":
            result = run_ultimate(cloner)
        elif sub == "3":
            result = run_two_pass(cloner)
        else:
            print("  ⚠️ 올바른 메뉴를 선택하세요.")
            return

        if result:
            logger.info(f"생성 완료: {result}")
            after_generate(result)
    except KeyboardInterrupt:
        print("\n⚠️ 취소됨")
    except Exception as e:
        print(f"\n❌ 에러: {e}")
        logger.error(f"에러: {e}")


# ─────────────────────────────────────────────
# 2. LoRA 적용 클로닝
# ─────────────────────────────────────────────
def menu_lora_clone(cloner, logger):
    print("\n  📌 LoRA 적용 클로닝:")
    print("  [1] 일반 적용")
    print("  [2] 적용 + 감정 제어")
    print("  [b] 돌아가기")

    sub = input("\n  선택 [기본=1]: ").strip()
    if not sub:
        sub = "1"
    if sub == "b":
        return

    lora_path = select_lora()
    if lora_path is None:
        return

    if cloner.current_lora != lora_path:
        print(f"\n⚠️ LoRA 모델 전환 (~20초 소요)")
    cloner.load_model(lora_path=lora_path)

    try:
        if sub == "1":
            result = run_lora_basic(cloner)
        elif sub == "2":
            result = run_lora_with_style(cloner)
        else:
            print("  ⚠️ 올바른 메뉴를 선택하세요.")
            return

        if result:
            logger.info(f"LoRA 생성 완료: {result}")
            after_generate(result)
    except KeyboardInterrupt:
        print("\n⚠️ 취소됨")
    except Exception as e:
        print(f"\n❌ 에러: {e}")
        logger.error(f"에러: {e}")


# ─────────────────────────────────────────────
# 3. LoRA 학습
# ─────────────────────────────────────────────
def menu_lora_train():
    from app.trainer import (
        create_config, run_training, list_lora_models,
        find_model_path, find_latest_checkpoint,
    )

    print("\n  📌 LoRA 학습:")
    print("  [1] 새 학습 시작")
    print("  [2] 학습된 모델 목록")
    print("  [b] 돌아가기")

    sel = input("  선택: ").strip()

    if sel == "b":
        return

    elif sel == "2":
        models = list_lora_models()
        if not models:
            print("\n  ⚠️ 학습된 모델이 없습니다.")
            return
        print(f"\n  📦 LoRA 모델 목록 ({len(models)}개):")
        print("  " + "-" * 40)
        for m in models:
            status = "✅" if m["has_weights"] else "⏳ 미완료"
            print(f"  · {m['name']} ({m['weight_count']}개 체크포인트) {status}")
        print("  " + "-" * 40)

    elif sel == "1":
        model_path = find_model_path()
        if not model_path:
            print("  ❌ VoxCPM2 모델을 찾을 수 없습니다.")
            return

        print(f"\n  모델 경로: {model_path}")

        output_name = input("  학습 이름 (예: ashe_lora): ").strip()
        if not output_name:
            print("  ⚠️ 이름을 입력하세요.")
            return

        # 이어하기 확인
        latest = find_latest_checkpoint(output_name)
        if latest:
            print(f"\n  📦 기존 체크포인트 발견: {output_name}")
            print("  [1] 이어서 학습 (Resume)")
            print("  [2] 처음부터 새로 학습")
            resume_sel = input("  선택 [기본=1]: ").strip()
            if not resume_sel:
                resume_sel = "1"
            if resume_sel == "2":
                latest = ""

        manifest = input("  Train Manifest 경로 [training_manifest.jsonl]: ").strip()
        if not manifest:
            manifest = "training_manifest.jsonl"
        if not Path(manifest).exists():
            print(f"  ❌ 파일 없음: {manifest}")
            return

        print(f"\n  ⚙️ 학습 파라미터 (Enter = 기본값)")
        lr = input("  Learning Rate [0.0001]: ").strip()
        lr = float(lr) if lr else 0.0001
        iters = input("  Max Iterations [5000]: ").strip()
        iters = int(iters) if iters else 5000
        save_int = input("  Save Interval [500]: ").strip()
        save_int = int(save_int) if save_int else 500

        config_path = create_config(
            output_name=output_name, train_manifest=manifest,
            model_path=model_path, learning_rate=lr,
            max_iterations=iters, save_interval=save_int,
        )

        if latest:
            print(f"  🔄 이어하기 모드: {latest}")

        print(f"  📄 설정 저장: {config_path}")
        confirm = input("  학습 시작? (y/n) [y]: ").strip()
        if confirm.lower() != "n":
            run_training(config_path)


# ─────────────────────────────────────────────
# 생성 후 동작
# ─────────────────────────────────────────────
def after_generate(result: list):
    if not result:
        return

    print("\n📌 다음 동작:")
    print("  [1] 첫 번째 결과 재생")
    print("  [2] 출력 폴더 열기")
    print("  [Enter] 건너뛰기")

    sel = input("  선택: ").strip()
    if sel == "1":
        play_audio(result[0])
    elif sel == "2":
        open_output_folder()


# ─────────────────────────────────────────────
# 공통 입력 함수
# ─────────────────────────────────────────────
def select_reference() -> str:
    files = print_audio_list()
    if not files:
        return None
    while True:
        sel = input("\n참조 음성 번호 선택 (또는 전체 경로 입력): ").strip()
        if sel.isdigit():
            idx = int(sel) - 1
            if 0 <= idx < len(files):
                print(f"  ✅ 선택: {files[idx].name}")
                return str(files[idx])
            else:
                print("  ⚠️ 범위 밖입니다.")
        elif sel:
            p = Path(sel)
            if p.exists():
                print(f"  ✅ 선택: {p.name}")
                return str(p)
            else:
                print(f"  ⚠️ 파일 없음: {sel}")


def select_lora() -> str:
    if not LORA_DIR.exists():
        print("  ⚠️ lora 폴더가 없습니다.")
        return None

    checkpoints = []
    for model_dir in sorted(LORA_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        ckpt_dir = model_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        for step_dir in sorted(ckpt_dir.iterdir()):
            if step_dir.is_dir() and (step_dir / "lora_weights.safetensors").exists():
                checkpoints.append({
                    "name": f"{model_dir.name}/{step_dir.name}",
                    "path": str(step_dir),
                })

    if not checkpoints:
        print("  ⚠️ 학습된 LoRA 체크포인트가 없습니다.")
        return None

    print(f"\n  📦 LoRA 체크포인트 ({len(checkpoints)}개):")
    print("  " + "-" * 40)
    for i, ck in enumerate(checkpoints, 1):
        print(f"  [{i}] {ck['name']}")
    print("  " + "-" * 40)

    sel = input("  선택: ").strip()
    if sel.isdigit():
        idx = int(sel) - 1
        if 0 <= idx < len(checkpoints):
            print(f"  ✅ 선택: {checkpoints[idx]['name']}")
            return checkpoints[idx]["path"]
    print("  ⚠️ 잘못된 선택")
    return None


def ask_parameters() -> dict:
    print(f"\n⚙️ 파라미터 설정 (Enter = 기본값)")
    print(f"   CFG: 높을수록 품질↑ 속도↓ (0=끄기, 추천: 2~5)")
    print(f"   Steps: 높을수록 품질↑ 속도↓ (추천: 5~15)")
    print(f"   ────────────────────────────")

    cfg_input = input(f"   CFG [{DEFAULT_CFG_VALUE}]: ").strip()
    try:
        cfg_value = float(cfg_input) if cfg_input else DEFAULT_CFG_VALUE
    except ValueError:
        cfg_value = DEFAULT_CFG_VALUE

    steps_input = input(f"   Steps [{DEFAULT_INFERENCE_TIMESTEPS}]: ").strip()
    try:
        steps = int(steps_input) if steps_input else DEFAULT_INFERENCE_TIMESTEPS
    except ValueError:
        steps = DEFAULT_INFERENCE_TIMESTEPS

    print(f"   ✅ CFG={cfg_value}, Steps={steps}")
    return {"cfg_value": cfg_value, "inference_timesteps": steps}


def ask_count() -> int:
    count_input = input(f"\n🔢 생성 개수 [기본=1]: ").strip()
    try:
        return max(1, int(count_input)) if count_input else 1
    except ValueError:
        return 1


def ask_style() -> str:
    print("\n🎨 스타일 지정 (Enter = 스타일 없이 진행):")
    print("   ────────────────────────────")
    for key, val in STYLE_PRESETS.items():
        print(f"   [{key:>2}] {val['name']}")
    print("   ────────────────────────────")

    sel = input("   선택 (Enter=없음): ").strip()

    if not sel:
        print("   ✅ 스타일 없이 진행")
        return ""
    elif sel == "0":
        print("   ────────────────────────────")
        print("   입력 예시 (영어, 조합 가능):")
        print("   · 감정: cheerful / sad / angry / excited / calm")
        print("   · 톤:   gentle / sweet / deep / soft / loud")
        print("   · 속도: slightly faster / slightly slower")
        print("   · 성별: young woman / mature male")
        print("   · 조합: slightly faster, cheerful tone")
        print("   ────────────────────────────")
        custom = input("   스타일 입력: ").strip()
        if custom:
            print(f"   ✅ 스타일: {custom}")
        return custom
    elif sel in STYLE_PRESETS:
        style = STYLE_PRESETS[sel]
        print(f"   ✅ 스타일: {style['name']} → {style['prompt']}")
        return style["prompt"]
    else:
        print("   ⚠️ 잘못된 선택. 스타일 없이 진행")
        return ""


def apply_style(text: str, style: str) -> str:
    if style:
        return f"({style}){text}"
    return text


def ask_transcript(cloner) -> tuple:
    print("\n📝 대본 처리 방식 선택:")
    print("  [1] 자동 추출 (Whisper)")
    print("  [2] 직접 입력")
    sub = input("  선택 [기본=1]: ").strip()
    if not sub:
        sub = "1"
    if sub == "2":
        print("\n참조 음성의 대본을 입력하세요:")
        prompt_text = input("  > ").strip()
        return prompt_text, False
    else:
        return None, True


def get_text_and_ref(cloner) -> tuple:
    ref_path = select_reference()
    if ref_path is None:
        return None, None
    print("\n💬 생성할 텍스트를 입력하세요:")
    text = input("  > ").strip()
    if not text:
        print("  ⚠️ 텍스트가 비어있습니다.")
        return None, None
    return ref_path, text


# ─────────────────────────────────────────────
# 모드별 실행 함수
# ─────────────────────────────────────────────
def run_controllable(cloner):
    ref_path, text = get_text_and_ref(cloner)
    if not ref_path:
        return None
    params = ask_parameters()
    count = ask_count()
    style = ask_style()
    styled_text = apply_style(text, style)
    return cloner.clone_voice(
        text=styled_text, reference_wav_path=ref_path, count=count, **params,
    )


def run_ultimate(cloner):
    ref_path, text = get_text_and_ref(cloner)
    if not ref_path:
        return None
    params = ask_parameters()
    count = ask_count()
    prompt_text, auto_transcribe = ask_transcript(cloner)
    return cloner.clone_voice(
        text=text, reference_wav_path=ref_path,
        prompt_text=prompt_text, ultimate=True,
        auto_transcribe=auto_transcribe, count=count, **params,
    )


def run_two_pass(cloner):
    ref_path, text = get_text_and_ref(cloner)
    if not ref_path:
        return None
    params = ask_parameters()
    count = ask_count()
    style = ask_style()
    prompt_text, auto_transcribe = ask_transcript(cloner)
    return cloner.two_pass_clone(
        text=text, reference_wav_path=ref_path, style=style,
        prompt_text=prompt_text, auto_transcribe=auto_transcribe,
        count=count, **params,
    )


def run_lora_basic(cloner):
    ref_path, text = get_text_and_ref(cloner)
    if not ref_path:
        return None
    params = ask_parameters()
    count = ask_count()
    return cloner.clone_voice(
        text=text, reference_wav_path=ref_path, count=count, **params,
    )


def run_lora_with_style(cloner):
    ref_path, text = get_text_and_ref(cloner)
    if not ref_path:
        return None
    params = ask_parameters()
    count = ask_count()
    style = ask_style()
    styled_text = apply_style(text, style)
    return cloner.clone_voice(
        text=styled_text, reference_wav_path=ref_path, count=count, **params,
    )


def run_voice_design(cloner):
    print("\n💬 생성할 텍스트를 입력하세요:")
    text = input("  > ").strip()
    if not text:
        print("  ⚠️ 텍스트가 비어있습니다.")
        return None

    print("\n🎨 음성 설명을 입력하세요 (영어 권장):")
    print("  예: A young woman, gentle and sweet voice")
    print("  예: mature male, deep and calm voice")
    print("  예: child, bright and energetic")
    voice_desc = input("  > ").strip()
    if not voice_desc:
        print("  ⚠️ 음성 설명이 비어있습니다.")
        return None

    params = ask_parameters()
    count = ask_count()

    return cloner.voice_design(
        text=text, voice_description=voice_desc, count=count, **params,
    )


if __name__ == "__main__":
    main()