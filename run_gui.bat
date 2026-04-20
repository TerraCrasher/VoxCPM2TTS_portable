@echo off
chcp 65001 >nul
title VoxCPM2 Voice Cloner - GUI
cd /d %~dp0

if not exist venv (
    echo ❌ 설치가 필요합니다. install.bat을 먼저 실행하세요.
    pause
    exit /b
)

call venv\Scripts\activate
python run_gui.py
pause