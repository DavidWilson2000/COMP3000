from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import shutil

# -----------------------------
# CONFIG
# -----------------------------
RAW_VIDEO_DIR = Path("data/raw_video")
RUNS_ROOT = Path("runs/video_runs")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

SRC_DIR = Path("src")
SCRIPT_EXTRACT = SRC_DIR / "extract_frames.py"
SCRIPT_SELECT = SRC_DIR / "select_frames_for_labeling.py"
SCRIPT_CROP = SRC_DIR / "crop_fish_from_frames.py"
SCRIPT_FILTER = SRC_DIR / "filter_crops_quality.py"
SCRIPT_DETECT_CLASSIFY = SRC_DIR / "detect_and_classify.py"
SCRIPT_CLUSTER = SRC_DIR / "cluster_fish.py"

# Models (set these to whatever you want the pipeline to use)
DET_MODEL = Path("runs/detect/runs/detect/fish_loop/weights/best.pt")
CLS_MODEL = Path("runs/classify/runs/species_cls_v2/weights/best.pt")


# Extraction / selection
EXTRACT_EVERY_N_FRAMES = 2
SELECT_MAX_FRAMES = 500

# Detect+classify source folder choice
DETECT_CLASSIFY_SOURCE_KIND = "selected"  # "selected" or "all"

# Crop settings
CROP_CONF = 0.25
CROP_IMGSZ = 640
CROP_PAD = 0.15
CROP_MIN_SIZE = 32
CROP_MAX_IMAGES = 2000

# Filter settings (tune for underwater)
FILTER_MIN_LONG_SIDE = 96
FILTER_MIN_AREA = 96 * 96
FILTER_MIN_SHARPNESS = 20.0


# Detect+classify settings
DC_CONF_DET = 0.25
DC_IMGSZ_DET = 640
DC_PAD = 0.15
DC_MIN_CROP = 64
DC_BLUR_THRESH = 50.0
DC_CONF_CLS = 0.65
DC_MARGIN = 0.15
DC_IMGSZ_CLS = 224
DC_LIMIT = 0  # 0 = no limit


# -----------------------------
# Helpers
# -----------------------------
def ensure_exists(path: Path, kind: str = "file") -> None:
    if kind == "file" and not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    if kind == "dir" and not path.is_dir():
        raise SystemExit(f"Missing directory: {path}")


def run_step(step_name: str, cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {step_name}")
    print("CMD:", " ".join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.stdout:
        print("\n--- STDOUT ---")
        print(result.stdout)

    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)

    if result.returncode != 0:
        raise SystemExit(f"\n❌ Step failed: {step_name} (exit code {result.returncode})")


def pick_next_video(raw_dir: Path) -> Path | None:
    # ignore already-processed folder
    vids = sorted([
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ])
    return vids[0] if vids else None


def make_run_dir(video_path: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_ROOT / f"{video_path.stem}__{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    print("Python executable:", sys.executable)

    ensure_exists(RAW_VIDEO_DIR, "dir")
    ensure_exists(SCRIPT_EXTRACT, "file")
    ensure_exists(SCRIPT_SELECT, "file")
    ensure_exists(SCRIPT_CROP, "file")
    ensure_exists(SCRIPT_FILTER, "file")
    ensure_exists(SCRIPT_DETECT_CLASSIFY, "file")
    ensure_exists(SCRIPT_CLUSTER, "file")
    ensure_exists(DET_MODEL, "file")
    ensure_exists(CLS_MODEL, "file")

    video_path = pick_next_video(RAW_VIDEO_DIR)
    if video_path is None:
        raise SystemExit(f"No videos found in {RAW_VIDEO_DIR} (extensions: {sorted(VIDEO_EXTS)})")

    # Per-video run folders
    run_dir = make_run_dir(video_path)
    frames_dir = run_dir / "frames"
    selected_dir = run_dir / "selected_frames"
    crops_dir = run_dir / "crops"
    crops_good_dir = run_dir / "crops_good"
    annotated_dir = run_dir / "annotated"
    clusters_dir = run_dir / "clusters"

    for d in [frames_dir, selected_dir, crops_dir, crops_good_dir, annotated_dir, clusters_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n🎬 Video: {video_path}")
    print(f"📁 Run dir: {run_dir}")

    # STEP 1: Extract frames
    run_step(
        "Extract frames",
        [
            sys.executable, str(SCRIPT_EXTRACT),
            "--video", str(video_path),
            "--out", str(frames_dir),
            "--every", str(EXTRACT_EVERY_N_FRAMES),
        ],
    )

    # STEP 2: Select frames
    run_step(
        "Select frames for labeling",
        [
            sys.executable, str(SCRIPT_SELECT),
            "--source", str(frames_dir),
            "--out", str(selected_dir),
            "--max", str(SELECT_MAX_FRAMES),
        ],
    )

    # STEP 3: Crop fish (detector crops)
    run_step(
        "Crop fish from frames",
        [
            sys.executable, str(SCRIPT_CROP),
            "--source", str(selected_dir),
            "--out", str(crops_dir),
            "--model", str(DET_MODEL),
            "--conf", str(CROP_CONF),
            "--imgsz", str(CROP_IMGSZ),
            "--pad", str(CROP_PAD),
            "--min_crop", str(CROP_MIN_SIZE),
            "--max_images", str(CROP_MAX_IMAGES),
        ],
    )

    # STEP 4: Filter crops quality
    run_step(
        "Filter crop quality",
        [
            sys.executable, str(SCRIPT_FILTER),
            "--source", str(crops_dir),
            "--out", str(crops_good_dir),
            "--min_long_side", str(FILTER_MIN_LONG_SIDE),
            "--min_area", str(FILTER_MIN_AREA),
            "--min_sharpness", str(FILTER_MIN_SHARPNESS),
        ],
    )

    # STEP 5: Detect + classify (annotate frames)
    det_source = selected_dir if DETECT_CLASSIFY_SOURCE_KIND == "selected" else frames_dir
    run_step(
        "Detect and classify",
        [
            sys.executable, str(SCRIPT_DETECT_CLASSIFY),
            "--source", str(det_source),
            "--out", str(annotated_dir),
            "--det_model", str(DET_MODEL),
            "--cls_model", str(CLS_MODEL),
            "--conf_det", str(DC_CONF_DET),
            "--imgsz_det", str(DC_IMGSZ_DET),
            "--pad", str(DC_PAD),
            "--min_crop", str(DC_MIN_CROP),
            "--blur_thresh", str(DC_BLUR_THRESH),
            "--conf_cls", str(DC_CONF_CLS),
            "--margin", str(DC_MARGIN),
            "--imgsz_cls", str(DC_IMGSZ_CLS),
            "--limit", str(DC_LIMIT),
        ],
    )

    # STEP 6: Cluster (batch, after video)
    run_step(
    "Cluster species (unsupervised)",
    [
        sys.executable, str(SCRIPT_CLUSTER),
        "--source", str(crops_good_dir),
        "--out", str(clusters_dir),
    ],
)


    # Move processed video to archive
    archive_dir = RAW_VIDEO_DIR / "_processed"
    archive_dir.mkdir(exist_ok=True)
    shutil.move(str(video_path), str(archive_dir / video_path.name))
    print(f"\n📦 Moved video to: {archive_dir / video_path.name}")

    print("\n✅ Done!")
    print("Run folder:", run_dir)
    print("Annotated:", annotated_dir)
    print("Clusters:", clusters_dir)


if __name__ == "__main__":
    main()
