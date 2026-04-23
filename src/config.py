from __future__ import annotations

import os
import sys
from pathlib import Path


def detect_project_root() -> Path:
    """
    Best-effort project root detection.

    Priority:
    1. FISH_AI_ROOT environment variable
    2. If running as a PyInstaller EXE, walk up from the EXE folder
    3. This file's parent folder, if it looks like the project root
    4. Common Windows default used in this project
    5. Current working directory
    """

    env_root = os.environ.get("FISH_AI_ROOT")
    if env_root:
        p = Path(env_root).expanduser()
        if p.exists():
            return p

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        for parent in [exe_dir, *exe_dir.parents]:
            if (parent / "src").exists() and (parent / "runs").exists():
                return parent
        if len(exe_dir.parents) >= 3:
            return exe_dir.parents[2]

    here = Path(__file__).resolve().parent

    if (here / "src").exists() and (here / "runs").exists():
        return here

    if (here.parent / "src").exists() and (here.parent / "runs").exists():
        return here.parent

    default_windows = Path(r"G:\fish_ai")
    if default_windows.exists():
        return default_windows

    cwd = Path.cwd()
    if (cwd / "src").exists() and (cwd / "runs").exists():
        return cwd

    if (cwd.parent / "src").exists() and (cwd.parent / "runs").exists():
        return cwd.parent

    return here


PROJECT_ROOT = detect_project_root()
SRC_DIR = PROJECT_ROOT / "src"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RUNS_DIR = PROJECT_ROOT / "runs"
DETECT_RUNS_DIR = RUNS_DIR / "detect"
VIDEO_RUNS_DIR = RUNS_DIR / "video_runs"
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "raw_video"
PROCESSED_VIDEO_DIR = RAW_VIDEO_DIR / "_processed"

TRAIN_SCRIPT = SRC_DIR / "train_detector_only.py"
PIPELINE_SCRIPT = SRC_DIR / "run_one_video.py"

DEFAULT_DATASET_YAML = DATASETS_DIR / "fish_dataset" / "data.yaml"
DEFAULT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

WINDOW_TITLE = "Fish AI Dashboard"
APP_MIN_SIZE = (1100, 760)
PREVIEW_SIZE = (420, 260)


def get_python_executable() -> Path:
    if DEFAULT_VENV_PYTHON.exists():
        return DEFAULT_VENV_PYTHON
    return DEFAULT_VENV_PYTHON
