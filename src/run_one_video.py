from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RAW_VIDEO_DIR = Path("data/raw_video")
RUNS_ROOT = Path("runs/video_runs")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
SRC_DIR = Path("src")

SCRIPT_EXTRACT = SRC_DIR / "extract_frames.py"
SCRIPT_SELECT = SRC_DIR / "select_frames_for_labeling.py"
SCRIPT_CROP = SRC_DIR / "crop_fish_from_frames.py"
SCRIPT_FILTER = SRC_DIR / "filter_crops_quality.py"
SCRIPT_CLUSTER = SRC_DIR / "cluster_fish.py"

DET_MODEL_CANDIDATES = [
    Path("runs/detect/fish_loop/weights/best.pt"),
    Path("runs/detect/runs/detect/fish_loop/weights/best.pt"),
    Path("runs/detect/fish_detector_v1/weights/best.pt"),
    Path("yolov8m.pt"),
    Path("yolov8n.pt"),
]

EXTRACT_EVERY_N_FRAMES = 2
SELECT_MAX_FRAMES = 1200
SELECT_MIN_SHARPNESS = 15.0
SELECT_MIN_SCENE_DELTA = 0.14
SELECT_BACKFILL = False
SELECT_MIN_FRAME_GAP = 12

CROP_CONF = 0.30
CROP_IMGSZ = 640
CROP_PAD = 0.10
CROP_MIN_SIZE = 40
CROP_MAX_IMAGES = 4000

FILTER_MIN_LONG_SIDE = 96
FILTER_MIN_AREA = 96 * 96
FILTER_MIN_SHARPNESS = 20.0
FILTER_MIN_CONTRAST = 12.0
FILTER_MIN_ASPECT_RATIO = 0.30
FILTER_MAX_ASPECT_RATIO = 3.50
FILTER_DEDUPE_HAMMING = 5

CLUSTER_USE_COLOR_HIST = True
CLUSTER_USE_SIZE_FEATS = True
CLUSTER_WHITE_BALANCE = True
CLUSTER_USE_UMAP = False
CLUSTER_UMAP_DIM = 32
CLUSTER_UMAP_NEIGHBORS = 10
CLUSTER_UMAP_MIN_DIST = 0.05
CLUSTER_MIN_CLUSTER_SIZE = 8
CLUSTER_MIN_SAMPLES = 5

ARCHIVE_PROCESSED_VIDEO = True


def ensure_exists(path: Path, kind: str = "file") -> None:
    if kind == "file" and not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    if kind == "dir" and not path.is_dir():
        raise SystemExit(f"Missing directory: {path}")


def resolve_first_existing(candidates: list[Path], model_name: str) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    joined = "\n - ".join(str(p) for p in candidates)
    raise SystemExit(f"Could not resolve {model_name}. Checked:\n - {joined}")


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
    videos = sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
    return videos[0] if videos else None


def make_run_dir(video_path: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_ROOT / f"{video_path.stem}__{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    print("Python executable:", sys.executable)

    ensure_exists(RAW_VIDEO_DIR, "dir")
    for script in [SCRIPT_EXTRACT, SCRIPT_SELECT, SCRIPT_CROP, SCRIPT_FILTER, SCRIPT_CLUSTER]:
        ensure_exists(script, "file")

    det_model = resolve_first_existing(DET_MODEL_CANDIDATES, "detector model")
    print(f"Resolved detector model:   {det_model}")

    video_path = pick_next_video(RAW_VIDEO_DIR)
    if video_path is None:
        raise SystemExit(f"No videos found in {RAW_VIDEO_DIR} (extensions: {sorted(VIDEO_EXTS)})")

    run_dir = make_run_dir(video_path)
    frames_dir = run_dir / "frames"
    selected_dir = run_dir / "selected_frames"
    crops_dir = run_dir / "crops"
    crops_good_dir = run_dir / "crops_good"
    clusters_dir = run_dir / "clusters"

    for d in [frames_dir, selected_dir, crops_dir, crops_good_dir, clusters_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n🎬 Video: {video_path}")
    print(f"📁 Run dir: {run_dir}")

    run_step(
        "Extract frames",
        [
            sys.executable, str(SCRIPT_EXTRACT),
            "--video", str(video_path),
            "--out", str(frames_dir),
            "--every", str(EXTRACT_EVERY_N_FRAMES),
        ],
    )

    select_cmd = [
        sys.executable, str(SCRIPT_SELECT),
        "--source", str(frames_dir),
        "--out", str(selected_dir),
        "--max", str(SELECT_MAX_FRAMES),
        "--min_sharpness", str(SELECT_MIN_SHARPNESS),
        "--min_scene_delta", str(SELECT_MIN_SCENE_DELTA),
        "--min_frame_gap", str(SELECT_MIN_FRAME_GAP),
    ]
    if SELECT_BACKFILL:
        select_cmd.append("--backfill")
    run_step("Select frames for labeling", select_cmd)

    run_step(
        "Crop fish from frames",
        [
            sys.executable, str(SCRIPT_CROP),
            "--source", str(selected_dir),
            "--out", str(crops_dir),
            "--model", str(det_model),
            "--conf", str(CROP_CONF),
            "--imgsz", str(CROP_IMGSZ),
            "--pad", str(CROP_PAD),
            "--min_crop", str(CROP_MIN_SIZE),
            "--max_images", str(CROP_MAX_IMAGES),
            "--csv_path", str(crops_dir / "crops.csv"),
        ],
    )

    run_step(
        "Filter crop quality",
        [
            sys.executable, str(SCRIPT_FILTER),
            "--source", str(crops_dir),
            "--out", str(crops_good_dir),
            "--min_long_side", str(FILTER_MIN_LONG_SIDE),
            "--min_area", str(FILTER_MIN_AREA),
            "--min_sharpness", str(FILTER_MIN_SHARPNESS),
            "--min_contrast", str(FILTER_MIN_CONTRAST),
            "--min_aspect_ratio", str(FILTER_MIN_ASPECT_RATIO),
            "--max_aspect_ratio", str(FILTER_MAX_ASPECT_RATIO),
            "--dedupe_hamming", str(FILTER_DEDUPE_HAMMING),
            "--report_csv", str(crops_good_dir / "filter_report.csv"),
        ],
    )

    cluster_cmd = [
        sys.executable, str(SCRIPT_CLUSTER),
        "--source", str(crops_good_dir),
        "--out", str(clusters_dir),
        "--min_cluster_size", str(CLUSTER_MIN_CLUSTER_SIZE),
        "--min_samples", str(CLUSTER_MIN_SAMPLES),
    ]
    if CLUSTER_USE_COLOR_HIST:
        cluster_cmd.append("--use_color_hist")
    if CLUSTER_USE_SIZE_FEATS:
        cluster_cmd.append("--use_size_feats")
    if CLUSTER_WHITE_BALANCE:
        cluster_cmd.append("--white_balance")
    if CLUSTER_USE_UMAP:
        cluster_cmd += [
            "--use_umap",
            "--umap_dim", str(CLUSTER_UMAP_DIM),
            "--umap_neighbors", str(CLUSTER_UMAP_NEIGHBORS),
            "--umap_min_dist", str(CLUSTER_UMAP_MIN_DIST),
        ]
    run_step("Cluster species (unsupervised)", cluster_cmd)

    if ARCHIVE_PROCESSED_VIDEO:
        archive_dir = RAW_VIDEO_DIR / "_processed"
        archive_dir.mkdir(exist_ok=True)
        shutil.move(str(video_path), str(archive_dir / video_path.name))
        print(f"\n📦 Moved video to: {archive_dir / video_path.name}")

    print("\n✅ Done!")
    print("Run folder:", run_dir)
    print("Clusters:", clusters_dir)


if __name__ == "__main__":
    main()