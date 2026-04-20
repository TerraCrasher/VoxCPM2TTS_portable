"""유틸리티 함수"""

import os
import logging
import subprocess
import platform
from pathlib import Path
from datetime import datetime

from app.config import LOG_DIR, INPUT_DIR, OUTPUT_DIR


def setup_logger(name: str = "voxcpm") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    log_file = LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def list_reference_audio() -> list:
    extensions = {".wav", ".mp3", ".flac", ".ogg"}
    files = []
    for f in INPUT_DIR.rglob("*"):
        if f.suffix.lower() in extensions:
            files.append(f)
    return sorted(files)


def print_audio_list():
    files = list_reference_audio()
    if not files:
        print(f"\n⚠️  참조 음성 파일이 없습니다.")
        print(f"   📁 여기에 넣으세요: {INPUT_DIR}")
        return files

    print(f"\n📂 참조 음성 목록 ({len(files)}개):")
    print("-" * 40)
    for i, f in enumerate(files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        try:
            rel = f.relative_to(INPUT_DIR)
        except ValueError:
            rel = f.name
        print(f"  [{i}] {rel} ({size_mb:.1f}MB)")
    print("-" * 40)
    return files


def open_output_folder():
    path = str(OUTPUT_DIR)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
    print(f"   📂 폴더 열림: {path}")


def play_audio(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"   ⚠️ 파일 없음: {path}")
        return

    if platform.system() == "Windows":
        os.startfile(str(path))
    elif platform.system() == "Darwin":
        subprocess.Popen(["afplay", str(path)])
    else:
        subprocess.Popen(["aplay", str(path)])
    print(f"   ▶ 재생: {path.name}")