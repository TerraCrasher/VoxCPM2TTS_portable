# 🎙️ VoxCPM2 TTS Portable V1.0

VoxCPM2 기반 음성 클로닝 & 생성 도구

## 📋 시스템 요구사항

- **OS**: Windows 10/11
- **Python**: 3.10 ~ 3.12
- **GPU**: NVIDIA (VRAM 8GB 이상 권장)
- **CUDA**: 12.0 이상
- **디스크**: ~10GB (모델 + 가상환경)

## 🚀 설치 방법

1. `install.bat` 더블클릭 (최초 1회)
2. 설치 완료까지 대기 (약 5~10분)

## 📖 사용법

### GUI (권장)
`run_gui.bat` 더블클릭 → 브라우저 자동 열림

### CLI
`run.bat` 더블클릭 → 터미널에서 사용

### 첫 실행 시
- 모델 자동 다운로드 (~5GB, 최초 1회)
- 로딩 시간 약 20초

## 🔊 기능

| 기능 | 설명 |
|------|------|
| **Controllable 클로닝** | 참조 음성 + 스타일 제어 |
| **Ultimate 클로닝** | 참조 음성 정밀 복제 |
| **Two-pass 클로닝** | 정밀도 + 스타일 |
| **Voice Design** | 참조 음성 없이 설명으로 생성 |
| **LoRA 학습** | 특정 화자 음성 학습 |
| **LoRA 적용** | 학습된 화자 음색으로 생성 |

## 📂 폴더 구조
VoxCPM2TTS_portable_V1.0/
├── inputs/reference_audio/  ← 참조 음성 파일
├── inputs/training_data/    ← LoRA 학습용 음성
├── outputs/                 ← 생성된 음성 파일
├── lora/                    ← LoRA 체크포인트
├── install.bat              ← 설치
├── run_gui.bat              ← GUI 실행
└── run.bat                  ← CLI 실행

## 🎨 스타일 제어 (영어)
감정: cheerful / sad / angry / excited / calm
톤:   gentle / sweet / deep / soft / loud
속도: slightly faster / slightly slower
성별: young woman / mature male
조합: slightly faster, cheerful tone

## 🎓 LoRA 학습 방법

1. `inputs/training_data/`에 음성 파일 넣기 (5~15초 클립)
2. CLI에서 `prepare_training.py` 실행 (대본 자동 추출)
3. GUI 또는 CLI에서 LoRA 학습 시작

## ⚠️ 주의사항

- GPU 메모리 8GB 필요 (다른 프로그램 종료 권장)
- GUI와 CLI 동시 실행 금지
- 첫 실행 시 모델 다운로드에 시간 소요