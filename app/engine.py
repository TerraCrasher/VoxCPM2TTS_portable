"""VoxCPM2 음성 클로닝 엔진"""

import time
import json
import gc
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import soundfile as sf

from app.config import (
    MODEL_ID, DEVICE, LOAD_DENOISER,
    DEFAULT_CFG_VALUE, DEFAULT_INFERENCE_TIMESTEPS,
    OUTPUT_DIR, OUTPUT_FORMAT,
)


class VoiceCloner:

    def __init__(self):
        self.model = None
        self.whisper_model = None
        self.is_loaded = False
        self.current_lora = None

    def _cleanup(self):
        """기존 모델 해제 + VRAM 정리"""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load_model(self, lora_path: str = None):
        if self.is_loaded and lora_path == self.current_lora:
            print("✅ 모델이 이미 로드되어 있습니다.")
            return

        # 기존 모델 정리
        if self.is_loaded:
            print("🔄 기존 모델 해제 중...")
            self._cleanup()

        print("=" * 50)
        print("🔄 VoxCPM2 모델 로딩 중...")
        print(f"   모델: {MODEL_ID}")
        print(f"   장치: {DEVICE}")
        if lora_path:
            print(f"   LoRA: {Path(lora_path).parent.parent.name}/{Path(lora_path).name}")
        print("=" * 50)

        import torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        start = time.time()

        try:
            from voxcpm import VoxCPM

            kwargs = {"load_denoiser": LOAD_DENOISER}

            if lora_path:
                lora_config_file = Path(lora_path) / "lora_config.json"
                if lora_config_file.exists():
                    with open(lora_config_file, "r") as f:
                        lora_json = json.load(f)
                    lora_dict = lora_json["lora_config"]
                    kwargs["lora_config"] = SimpleNamespace(**lora_dict)
                    kwargs["lora_weights_path"] = lora_path
                else:
                    print(f"⚠️ lora_config.json 없음: {lora_config_file}")

            self.model = VoxCPM.from_pretrained(MODEL_ID, **kwargs)

            elapsed = time.time() - start
            self.is_loaded = True
            self.current_lora = lora_path
            print(f"✅ 모델 로드 완료! ({elapsed:.1f}초)")

        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("   재시도하려면 프로그램을 다시 시작하세요.")
            self._cleanup()
            raise

    def load_whisper(self):
        if self.whisper_model is not None:
            return
        print("\n🔄 Whisper 모델 로딩 중 (CPU)...")
        start = time.time()
        import whisper
        self.whisper_model = whisper.load_model("base", device="cpu")
        elapsed = time.time() - start
        print(f"✅ Whisper 로드 완료! ({elapsed:.1f}초)")

    def transcribe(self, audio_path: str) -> str:
        self.load_whisper()
        print(f"\n📝 대본 자동 추출 중: {Path(audio_path).name}")
        start = time.time()
        result = self.whisper_model.transcribe(audio_path)
        text = result["text"].strip()
        lang = result.get("language", "unknown")
        elapsed = time.time() - start
        print(f"   언어: {lang}")
        print(f"   대본: {text}")
        print(f"   소요: {elapsed:.1f}초")

        # Whisper 해제 (VRAM/RAM 절약)
        del self.whisper_model
        self.whisper_model = None
        gc.collect()

        return text

    def _generate_once(self, generate_kwargs: dict, progress_callback=None, cancel_check=None) -> tuple:
        import torch
        import sys
        import io
        import re
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

        class TqdmCapture(io.StringIO):
            def __init__(self, callback):
                super().__init__()
                self.callback = callback

            def write(self, s):
                if self.callback and s.strip():
                    match = re.search(r'(\d+)%', s)
                    if match:
                        pct = int(match.group(1))
                        self.callback(pct)
                return len(s)

            def flush(self):
                pass

        old_stderr = sys.stderr
        sys.stderr = TqdmCapture(progress_callback)

        try:
            with torch.inference_mode():
                start = time.time()

                if cancel_check:
                    # 별도 스레드에서 실행, 주기적으로 취소 확인
                    result = [None]
                    error = [None]

                    def run_generate():
                        try:
                            result[0] = self.model.generate(**generate_kwargs)
                        except Exception as e:
                            error[0] = e

                    thread = __import__('threading').Thread(target=run_generate)
                    thread.start()

                    while thread.is_alive():
                        thread.join(timeout=0.5)
                        if cancel_check():
                            # 취소 요청됨 → CUDA 중단
                            torch.cuda.empty_cache()
                            raise InterruptedError("사용자 취소")

                    if error[0]:
                        raise error[0]

                    wav = result[0]
                else:
                    wav = self.model.generate(**generate_kwargs)

                gen_time = time.time() - start
        finally:
            sys.stderr = old_stderr

        torch.cuda.empty_cache()
        return wav, gen_time

        try:
            with torch.inference_mode():
                start = time.time()
                wav = self.model.generate(**generate_kwargs)
                gen_time = time.time() - start
        finally:
            sys.stderr = old_stderr

        torch.cuda.empty_cache()
        return wav, gen_time

    def clone_voice(
        self,
        text: str,
        reference_wav_path: str,
        prompt_text: str = None,
        cfg_value: float = DEFAULT_CFG_VALUE,
        inference_timesteps: int = DEFAULT_INFERENCE_TIMESTEPS,
        ultimate: bool = False,
        auto_transcribe: bool = False,
        count: int = 1,
    ) -> list:
        if not self.is_loaded:
            self.load_model()

        ref_path = Path(reference_wav_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"❌ 참조 음성 파일 없음: {ref_path}")

        if ultimate and auto_transcribe and not prompt_text:
            prompt_text = self.transcribe(str(ref_path))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if ultimate and prompt_text:
            mode = "🎙️ Ultimate Cloning"
        elif self.current_lora:
            mode = "🎓 LoRA Cloning"
        else:
            mode = "🎛️ Controllable Cloning"

        print("\n" + "=" * 50)
        print(f"  모드: {mode}")
        print(f"  텍스트: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  참조음성: {ref_path.name}")
        if ultimate and prompt_text:
            print(f"  대본: {prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}")
        if self.current_lora:
            print(f"  LoRA: {Path(self.current_lora).name}")
        print(f"  CFG: {cfg_value} | Steps: {inference_timesteps}")
        print(f"  생성 개수: {count}")
        print("=" * 50)

        generate_kwargs = {
            "text": text,
            "reference_wav_path": str(ref_path),
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
        }
        if ultimate and prompt_text:
            generate_kwargs["prompt_wav_path"] = str(ref_path)
            generate_kwargs["prompt_text"] = prompt_text

        saved_files = []
        for i in range(count):
            print(f"\n🔊 [{i+1}/{count}] 생성 중...")
            wav, gen_time = self._generate_once(generate_kwargs)

            if count == 1:
                fname = f"clone_{timestamp}.{OUTPUT_FORMAT}"
            else:
                fname = f"clone_{timestamp}_{i+1:02d}.{OUTPUT_FORMAT}"

            output_path = OUTPUT_DIR / fname
            sf.write(str(output_path), wav, self.model.tts_model.sample_rate)
            saved_files.append(str(output_path))
            print(f"   ✅ 저장: {fname} ({gen_time:.1f}초)")

        print(f"\n✅ 완료! 총 {count}개 생성됨")
        print(f"   📁 폴더: {OUTPUT_DIR}")
        return saved_files

    def clone_voice_with_style(self, text, reference_wav_path, style="", **kwargs):
        if style:
            styled_text = f"({style}){text}"
        else:
            styled_text = text
        return self.clone_voice(text=styled_text, reference_wav_path=reference_wav_path, **kwargs)

    def two_pass_clone(
        self, text, reference_wav_path, style="",
        prompt_text=None, auto_transcribe=False,
        cfg_value=DEFAULT_CFG_VALUE, inference_timesteps=DEFAULT_INFERENCE_TIMESTEPS,
        count=1,
    ) -> list:
        if not self.is_loaded:
            self.load_model()

        print("\n" + "=" * 50)
        print("  🔁 Two-pass 클로닝")
        print("=" * 50)

        print("\n📌 [Pass 1/2] Ultimate 클로닝 (정밀 복제)")
        temp_results = self.clone_voice(
            text=text, reference_wav_path=reference_wav_path,
            prompt_text=prompt_text, ultimate=True,
            auto_transcribe=auto_transcribe,
            cfg_value=cfg_value, inference_timesteps=inference_timesteps,
            count=1,
        )
        temp_wav = temp_results[0]

        if not style:
            print("\n✅ 스타일 없음 → 1단계 결과를 최종 출력으로 사용")
            return temp_results

        print(f"\n📌 [Pass 2/2] 스타일 적용: {style}")
        final_results = self.clone_voice_with_style(
            text=text, reference_wav_path=temp_wav, style=style,
            cfg_value=cfg_value, inference_timesteps=inference_timesteps,
            count=count,
        )

        # Two-pass 임시파일 정리
        import os
        try:
            os.remove(temp_wav)
            print(f"   🗑️ 임시파일 삭제: {Path(temp_wav).name}")
        except OSError:
            pass

        print("\n🎉 Two-pass 클로닝 완료!")
        return final_results

def voice_design(
        self,
        text: str,
        voice_description: str,
        cfg_value: float = DEFAULT_CFG_VALUE,
        inference_timesteps: int = DEFAULT_INFERENCE_TIMESTEPS,
        count: int = 1,
        progress_callback=None,
        cancel_check=None,
    ) -> list:
        """Voice Design - 설명만으로 목소리 생성"""
        if not self.is_loaded:
            self.load_model()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        designed_text = f"({voice_description}){text}"

        print("\n" + "=" * 50)
        print(f"  모드: 🎨 Voice Design")
        print(f"  텍스트: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  음성 설명: {voice_description[:50]}")
        print(f"  CFG: {cfg_value} | Steps: {inference_timesteps}")
        print(f"  생성 개수: {count}")
        print("=" * 50)

        generate_kwargs = {
            "text": designed_text,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
        }

        saved_files = []
        for i in range(count):
            print(f"\n🎨 [{i+1}/{count}] 생성 중...")
            wav, gen_time = self._generate_once(
                generate_kwargs,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

            if count == 1:
                fname = f"design_{timestamp}.{OUTPUT_FORMAT}"
            else:
                fname = f"design_{timestamp}_{i+1:02d}.{OUTPUT_FORMAT}"

            output_path = OUTPUT_DIR / fname
            sf.write(str(output_path), wav, self.model.tts_model.sample_rate)
            saved_files.append(str(output_path))
            print(f"   ✅ 저장: {fname} ({gen_time:.1f}초)")

        print(f"\n✅ 완료! 총 {count}개 생성됨")
        return saved_files