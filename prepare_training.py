"""학습 데이터 준비 - Whisper로 대본 자동 추출 + jsonl 생성"""

import json
import time
from pathlib import Path
from tqdm import tqdm
import whisper
import soundfile as sf


# ── 설정 ──
INPUT_DIR = Path("inputs/training_data")
OUTPUT_JSONL = Path("training_manifest.jsonl")
LANGUAGE = "ja"  # 일본어

def main():
    # 음성 파일 검색
    extensions = {".wav", ".mp3", ".flac", ".ogg"}
    audio_files = sorted([f for f in INPUT_DIR.iterdir() if f.suffix.lower() in extensions])

    if not audio_files:
        print(f"❌ 음성 파일이 없습니다: {INPUT_DIR}")
        print(f"   이 폴더에 wav 파일을 넣어주세요.")
        return

    print(f"\n📂 음성 파일 {len(audio_files)}개 발견")
    print("=" * 50)

    # Whisper 로드
    print("🔄 Whisper 모델 로딩 중...")
    start = time.time()
    model = whisper.load_model("base")
    print(f"✅ Whisper 로드 완료! ({time.time() - start:.1f}초)")

    # 대본 추출 + jsonl 생성
    entries = []
    errors = []

    print(f"\n📝 대본 추출 시작 (언어: {LANGUAGE})")
    print("=" * 50)

    for audio_file in tqdm(audio_files, desc="🎙️ 추출 진행", ncols=70):
        try:
            # 음성 길이 확인
            info = sf.info(str(audio_file))
            duration = round(info.duration, 2)

            # 너무 짧거나 긴 파일 스킵
            if duration < 0.5:
                print(f"   ⚠️ 스킵 (너무 짧음 {duration}초): {audio_file.name}")
                continue
            if duration > 30.0:
                print(f"   ⚠️ 스킵 (너무 김 {duration}초): {audio_file.name}")
                continue

            # Whisper 추출
            result = model.transcribe(
                str(audio_file),
                language=LANGUAGE,
            )
            text = result["text"].strip()

            if not text:
                print(f"   ⚠️ 스킵 (텍스트 없음): {audio_file.name}")
                continue

            entry = {
                "audio": str(audio_file.resolve()),
                "text": text,
                "duration": duration,
            }
            entries.append(entry)

            print(f"   ✅ [{duration:.1f}초] {audio_file.name}")
            print(f"      → {text[:60]}{'...' if len(text) > 60 else ''}")

        except Exception as e:
            errors.append((audio_file.name, str(e)))
            print(f"   ❌ 에러: {audio_file.name} → {e}")

    # jsonl 저장
    if entries:
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print("\n" + "=" * 50)
        print(f"✅ 완료!")
        print(f"   📄 jsonl 저장: {OUTPUT_JSONL.resolve()}")
        print(f"   📊 총 {len(entries)}개 클립")
        total_duration = sum(e["duration"] for e in entries)
        print(f"   ⏱️  총 길이: {total_duration:.1f}초 ({total_duration/60:.1f}분)")
    else:
        print("\n❌ 추출된 데이터가 없습니다.")

    if errors:
        print(f"\n⚠️ 에러 {len(errors)}건:")
        for name, err in errors:
            print(f"   · {name}: {err}")


if __name__ == "__main__":
    main()