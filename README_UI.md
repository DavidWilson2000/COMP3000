# Fish AI Dashboard UI

This is a lightweight Windows desktop dashboard for your fish AI project.

## What it does

- Launches `train_detector_only.py`
- Launches `run_one_video.py`
- Opens your dataset folder, runs folder, latest run folder, latest weights folder
- Reads the newest `results.csv` under `runs/detect`
- Shows the latest Precision, Recall, mAP50, and mAP50-95
- Previews `results.png` and `confusion_matrix.png`

## Files

- `app.py` - main Tkinter app
- `config.py` - project paths and app settings
- `metrics_reader.py` - reads latest metrics from YOLO results
- `ui_helpers.py` - subprocess and file-opening helpers
- `requirements.txt` - packages for the UI and EXE build
- `build_exe.bat` - Windows build script using PyInstaller

## How to use it

1. Copy all UI files into your `G:\fish_ai` project root.
2. Install the UI requirements:
   - `py -m pip install -r requirements.txt`
3. Run the app:
   - `py app.py`

If your project is not at `G:\fish_ai`, set an environment variable before launch:

```bat
set FISH_AI_ROOT=D:\path\to\your\fish_ai
py app.py
```

## How to build an EXE

1. Double-click `build_exe.bat`
2. When the build finishes, the EXE will be here:
   - `dist\FishAIDashboard\FishAIDashboard.exe`

## Notes

- The UI looks for your project `.venv\Scripts\python.exe` first.
- If that does not exist, it falls back to the Python interpreter running the app.
- The app does not replace your scripts. It just wraps the scripts you already use.
- The dashboard reads the newest `results.csv` found anywhere under `runs/detect`, so it still works even if nested `runs/detect/runs/detect/...` folders exist.
