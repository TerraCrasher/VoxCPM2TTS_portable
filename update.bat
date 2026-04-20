@echo off
chcp 65001 >nul
title VoxCPM2 TTS - 업데이트
cd /d %~dp0

echo ══════════════════════════════════════════
echo   🔄 VoxCPM2 TTS - 업데이트
echo ══════════════════════════════════════════
echo.

:: Git 확인
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git이 설치되어 있지 않습니다.
    echo.
    echo    설치: https://git-scm.com/downloads
    echo.
    echo    또는 수동 업데이트:
    echo    https://github.com/TerraCrasher/VoxCPM2TTS_portable/releases
    echo    에서 최신 ZIP 다운로드 후 덮어쓰기
    echo.
    pause
    exit /b
)

:: Git 초기화 확인
if not exist .git (
    echo 🔄 Git 초기화 중...
    git init
    git remote add origin https://github.com/TerraCrasher/VoxCPM2TTS_portable.git
    git fetch origin main
    git reset --hard origin/main
    echo.
    echo ✅ 초기화 완료!
    goto :UPDATE_DEPS
)

echo [1/3] 사용자 데이터 보호 중...
git stash 2>nul

echo [2/3] 최신 버전 다운로드 중...
git fetch origin main
if errorlevel 1 (
    echo ❌ 서버 연결 실패. 인터넷 연결을 확인하세요.
    pause
    exit /b
)
git reset --hard origin/main

echo [3/3] 사용자 데이터 복원...
git stash pop 2>nul

:UPDATE_DEPS
echo.
echo 📦 의존성 업데이트 확인...
if exist venv (
    call venv\Scripts\activate
    pip install -r requirements.txt --quiet 2>nul
)

echo.
echo ══════════════════════════════════════════
echo   ✅ 업데이트 완료!
echo ══════════════════════════════════════════
echo.
echo   ⚠️ 사용자 데이터 (inputs, outputs, lora)는
echo      유지됩니다.
echo.
echo   run_gui.bat 으로 실행하세요.
echo ══════════════════════════════════════════
pause