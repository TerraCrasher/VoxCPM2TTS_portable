"""LoRA 학습 모듈 - 로컬 통합"""

import sys
import subprocess
import yaml
from pathlib import Path

from app.config import BASE_DIR, LORA_DIR


TRAIN_SCRIPT = BASE_DIR / "scripts" / "train_voxcpm_finetune.py"
MODEL_PATH = Path.home() / ".cache" / "huggingface" / "hub" / "models--openbmb--VoxCPM2" / "snapshots"


def get_venv_python() -> str:
    """가상환경 Python 경로 확인"""
    venv_python = BASE_DIR / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    # fallback
    return sys.executable


def find_model_path() -> str:
    if MODEL_PATH.exists():
        snapshots = list(MODEL_PATH.iterdir())
        if snapshots:
            return str(snapshots[0])
    return ""


def create_config(
    output_name: str,
    train_manifest: str,
    model_path: str = None,
    learning_rate: float = 0.0001,
    max_iterations: int = 5000,
    batch_size: int = 1,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    save_interval: int = 500,
) -> Path:
    if model_path is None:
        model_path = find_model_path()

    output_dir = LORA_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "pretrained_path": model_path,
        "train_manifest": train_manifest,
        "val_manifest": "",
        "save_path": str(output_dir / "checkpoints"),
        "tensorboard": str(output_dir / "logs"),
        "learning_rate": learning_rate,
        "max_steps": max_iterations,
        "num_iters": max_iterations,
        "batch_size": batch_size,
        "save_interval": save_interval,
        "valid_interval": 1000,
        "log_interval": 10,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "grad_accum_steps": 1,
        "num_workers": 2,
        "sample_rate": 16000,
        "out_sample_rate": 48000,
        "lora": {
            "r": lora_rank,
            "alpha": lora_alpha,
            "dropout": 0.0,
            "enable_lm": True,
            "enable_dit": True,
            "enable_proj": False,
            "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
        "lambdas": {
            "loss/diff": 1.0,
            "loss/stop": 1.0,
        },
    }

def find_latest_checkpoint(output_name: str) -> str:
    """기존 체크포인트에서 이어하기 위한 latest 경로 탐색"""
    ckpt_dir = LORA_DIR / output_name / "checkpoints" / "latest"
    if ckpt_dir.exists():
        return str(ckpt_dir)
    return ""

    config_path = output_dir / "train_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_training(config_path: Path):
    python_path = get_venv_python()

    print("\n" + "=" * 50)
    print("  🎓 LoRA 학습 시작")
    print(f"  설정: {config_path}")
    print(f"  Python: {python_path}")
    print("=" * 50)
    print("  (중단: Ctrl+C, 체크포인트는 자동 저장됩니다)")
    print()

    cmd = [
        python_path,
        str(TRAIN_SCRIPT),
        "--config_path", str(config_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(BASE_DIR),
    )

    try:
        for line in process.stdout:
            print(f"  {line}", end="")
        process.wait()

        if process.returncode == 0:
            print("\n✅ 학습 완료!")
        else:
            print(f"\n❌ 학습 실패 (코드: {process.returncode})")

    except KeyboardInterrupt:
        process.terminate()
        print("\n⚠️ 학습 중단됨 (저장된 체크포인트는 유지됩니다)")

    return process.returncode


def list_lora_models() -> list:
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for d in LORA_DIR.iterdir():
        if d.is_dir():
            weights = list(d.glob("**/*.safetensors")) + list(d.glob("**/*.pt"))
            models.append({
                "name": d.name,
                "path": str(d),
                "has_weights": len(weights) > 0,
                "weight_count": len(weights),
            })
    return models