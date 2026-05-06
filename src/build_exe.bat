@echo off
cd /d "%~dp0"

echo Cleaning old build folders...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo Building FishAIDashboard...
python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --windowed ^
  --name FishAIDashboard ^
  --paths "%CD%" ^
  --hidden-import=config ^
  --hidden-import=metrics_reader ^
  --hidden-import=ui_helpers ^
  app.py

echo.
echo Build complete.
echo Run the dashboard from:
echo src\dist\FishAIDashboard\FishAIDashboard.exe
pause