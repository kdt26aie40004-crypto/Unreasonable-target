@echo off
cd /d "%~dp0"
where pythonw >nul 2>nul
if errorlevel 1 (
  echo Python が見つかりません。https://www.python.org/ からインストールしてください。
  pause
  exit /b
)
start "" pythonw gui.py