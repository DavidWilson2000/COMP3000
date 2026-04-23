@echo off
setlocal
cd /d %~dp0

echo Building Fish AI Dashboard...
py -m pip install -r requirements.txt
py -m PyInstaller --noconfirm --windowed --name FishAIDashboard app.py

echo.
echo Build complete.
echo EXE location: dist\FishAIDashboard\FishAIDashboard.exe
pause
