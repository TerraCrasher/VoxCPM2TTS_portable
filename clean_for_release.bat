@echo off
chcp 65001 >nul
title 배포 정리
cd /d %~dp0

echo ══════════════════════════════════════════
echo   🧹 배포 전 정리
echo ══════════════════════════════════════════
echo.
echo   삭제 대상:
echo     - venv/ (가상환경)
echo     - __pycache__/ (캐시)
echo     - logs/*.log
echo     - outputs/*.wav
echo     - training_manifest.jsonl
echo.

set /p confirm=정리 시작? (y/n): 
if /i not "%confirm%"=="y" exit /b

echo.
echo [1/5] venv 삭제 중...
if exist venv rmdir /s /q venv

echo [2/5] __pycache__ 삭제 중...
for /d /r %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d"

echo [3/5] 로그 정리...
del /q logs\*.log 2>nul

echo [4/5] 출력 파일 정리...
del /q outputs\*.wav 2>nul

echo [5/5] 기타 정리...
del /q training_manifest.jsonl 2>nul

echo.
echo ✅ 정리 완료! 이제 폴더를 압축하여 배포하세요.
pause