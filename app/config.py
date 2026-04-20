"""설정 관리"""

from pathlib import Path

# ── 버전 ──
VERSION = "1.0.0"
GITHUB_REPO = "TerraCrasher/VoxCPM2TTS_portable"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}"

# ── 경로 설정 ──
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "inputs" / "reference_audio"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
LORA_DIR = BASE_DIR / "lora"
JSONL_DIR = BASE_DIR / "inputs" / "jsonl"

for d in [INPUT_DIR, OUTPUT_DIR, LOG_DIR, LORA_DIR, JSONL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 모델 설정 ──
MODEL_ID = "openbmb/VoxCPM2"
DEVICE = "cuda"
LOAD_DENOISER = False

# ── 음성 생성 기본값 ──
DEFAULT_CFG_VALUE = 5.0
DEFAULT_INFERENCE_TIMESTEPS = 15

# ── 출력 설정 ──
OUTPUT_FORMAT = "wav"

# ── 스타일 프리셋 ──
STYLE_PRESETS = {
    "0": {"name": "직접 입력", "prompt": ""},
    "1": {"name": "빠르게", "prompt": "slightly faster"},
    "2": {"name": "느리게", "prompt": "slightly slower"},
    "3": {"name": "밝고 활기찬", "prompt": "cheerful tone"},
    "4": {"name": "차분하고 부드러운", "prompt": "gentle and soft tone"},
    "5": {"name": "젊은 여성, 부드럽고 달콤한", "prompt": "A young woman, gentle and sweet voice"},
    "6": {"name": "빠르고 밝은", "prompt": "slightly faster, cheerful tone"},
}