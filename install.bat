@echo off
chcp 65001 >nul
title VoxCPM2 TTS - 설치
cd /d %~dp0

echo ══════════════════════════════════════════
echo   🎙️ VoxCPM2 TTS Portable - 설치
echo ══════════════════════════════════════════
echo.

:: Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되어 있지 않습니다.
    echo    Python 3.10~3.12를 설치해주세요.
    echo    https://www.python.org/downloads/
    pause
    exit /b
)

echo [1/4] 가상환경 생성 중...
python -m venv venv
if errorlevel 1 (
    echo ❌ 가상환경 생성 실패
    pause
    exit /b
)

echo [2/4] 가상환경 활성화...
call venv\Scripts\activate

echo [3/4] PyTorch CUDA 설치 중... (시간이 걸립니다)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo ⚠️ CUDA PyTorch 설치 실패. CPU 버전으로 설치합니다.
    pip install torch torchaudio
)

echo [4/4] 의존성 설치 중...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ 의존성 설치 실패
    pause
    exit /b
)

echo.
echo ══════════════════════════════════════════
echo   ✅ 설치 완료!
echo ══════════════════════════════════════════
echo.
echo   사용법:
echo     run_gui.bat  → GUI 실행 (브라우저)
echo     run.bat      → CLI 실행 (터미널)
echo.
echo   첫 실행 시 모델 자동 다운로드 (~5GB)
echo ══════════════════════════════════════════
pause