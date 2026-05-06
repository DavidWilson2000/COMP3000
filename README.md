# Smart Aqua Fish Detection Pipeline

Smart Aqua is a computer vision system for processing aquarium/fish video footage. It provides an end-to-end workflow for:

1. extracting frames from video,
2. selecting useful frames,
3. detecting fish using YOLOv8,
4. cropping detected fish,
5. filtering low-quality crops,
6. clustering visually similar fish crops,
7. reviewing outputs through a desktop dashboard.

The project is designed as a lightweight fish video analysis workflow. The clustering stage groups visually similar fish crops, but it should not be treated as confirmed species identification.

---

# Important: Downloading the Project Correctly

This project uses **Git LFS** for large files such as:

```text
runs/detect/fish_loop/weights/best.pt
data/raw_video/sample_video.mp4
```

Do **not** use “Download ZIP” unless you are sure GitHub has included the full LFS files.

The safest way to download the project is:

```bash
git lfs install
git clone https://github.com/DavidWilson2000/COMP3000.git
cd COMP3000
git lfs pull
```

After cloning, check that the model file exists and is not tiny:

```text
runs/detect/fish_loop/weights/best.pt
```

The real `best.pt` should be around 100MB+. If it is about 1KB, Git LFS has not downloaded the real model. Run:

```bash
git lfs pull
git lfs checkout
```

---

# Recommended Setup on a New Windows PC

## 1. Install Python

Install Python **3.11 or 3.12**.

During installation, tick:

```text
Add Python to PATH
```

Check Python is installed:

```bash
python --version
```

---

## 2. Open the Project Root

Open Command Prompt or PowerShell in the main project folder.

The project root should contain:

```text
src/
runs/
data/
datasets/
Documents/
README.md
```

Example:

```bash
cd C:\Users\YourName\Downloads\COMP3000
```

---

## 3. Create a Virtual Environment

From the project root, run:

```bash
python -m venv .venv
```

### If using Command Prompt

```bat
.venv\Scripts\activate.bat
```

### If using PowerShell

If PowerShell blocks activation with a security error, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate:

```powershell
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of the terminal line.

---

## 4. Install Requirements

From the project root, with the virtual environment activated, run:

```bash
pip install -r src\requirements.txt
```

This installs the packages needed for the detector, pipeline, clustering, and dashboard scripts.

If `cv2` is missing, install OpenCV manually:

```bash
pip install opencv-python
```

---

# Building the Dashboard UI

The dashboard executable is built using:

```text
src/build_exe.bat
```

To build it:

```bash
cd src
build_exe.bat
```

After the build finishes, open:

```text
src/dist/FishAIDashboard/
```

Run:

```text
FishAIDashboard.exe
```

The correct folder structure is:

```text
src/dist/FishAIDashboard/
    FishAIDashboard.exe
    _internal/
```

Do **not** move `FishAIDashboard.exe` away from `_internal`.

If you want to run it from the desktop, create a shortcut to:

```text
src/dist/FishAIDashboard/FishAIDashboard.exe
```

Do **not** run the EXE from:

```text
src/build/
```

The `build` folder is temporary and can cause missing Python DLL errors.

---

# Running the Dashboard

Once the requirements are installed and the dashboard has been built:

1. Open:

```text
src/dist/FishAIDashboard/
```

2. Double-click:

```text
FishAIDashboard.exe
```

3. Click:

```text
Refresh Summary
```

4. Add a video to:

```text
data/raw_video/
```

5. Click:

```text
Run Video Pipeline
```

The dashboard allows users to:

- run the video pipeline,
- train the detector,
- open raw video folders,
- open processed video folders,
- open latest video run folders,
- open dataset folders,
- open detector run folders,
- view latest metrics,
- open the results graph,
- open the confusion matrix.

---

# Running the Video Pipeline Without the Dashboard

From the project root:

```bash
.venv\Scripts\activate.bat
python src\run_one_video.py
```

Or in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python src\run_one_video.py
```

Before running, make sure there is a video inside:

```text
data/raw_video/
```

Supported video formats:

```text
.mp4
.mov
.avi
.mkv
```

---

# Project Workflow

The full pipeline is controlled by:

```text
src/run_one_video.py
```

The workflow is:

```text
Raw video
   ↓
Extract frames
   ↓
Select useful frames
   ↓
Detect and crop fish
   ↓
Filter crop quality
   ↓
Cluster similar fish crops
   ↓
Save final outputs
```

Each processed video creates a timestamped folder inside:

```text
runs/video_runs/
```

Example:

```text
runs/video_runs/FishFullA__2026-05-06_13-45-41/
```

---

# Main Output Folders

Each video run may contain:

```text
frames/
selected_frames/
crops/
crops_good/
clusters/
```

## Output meaning

| Folder/File | Purpose |
|---|---|
| `frames/` | All extracted video frames. |
| `selected_frames/` | Clearer and more useful frames selected for detection. |
| `crops/` | Raw fish crops created from YOLO detections. |
| `crops/crops.csv` | Metadata for detected crops. |
| `crops_good/` | Crops that passed quality filtering. |
| `crops_good/filter_report.csv` | Shows which crops were kept or rejected and why. |
| `clusters/` | Fish crops grouped into visually similar folders. |
| `clusters/clusters_summary.csv` | Number of crops in each cluster. |
| `clusters/cluster_map.csv` | Maps original crop paths to cluster output paths. |

---

# Detection Model

The final trained detector should be located at:

```text
runs/detect/fish_loop/weights/best.pt
```

This file is required for the crop extraction stage.

If the file is missing or only 1KB, Git LFS has not downloaded it properly. Run:

```bash
git lfs pull
git lfs checkout
```

If the model is still missing, the detector can be retrained using:

```bash
python src\train_detector_only.py
```

or by pressing:

```text
Train Detector
```

inside the dashboard.

---

# Dataset Location

The YOLO dataset should be located at:

```text
datasets/fish_dataset/
```

The dataset YAML should be:

```text
datasets/fish_dataset/data.yaml
```

The detector was trained as a single-class detector using the class:

```text
Fish
```

---

# Main Source Files

```text
src/app.py                         Dashboard UI
src/run_one_video.py               Full video pipeline runner
src/train_detector_only.py         YOLOv8 detector training script
src/extract_frames.py              Frame extraction
src/select_frames_for_labeling.py  Frame selection
src/crop_fish_from_frames.py       Fish detection and crop extraction
src/filter_crops_quality.py        Crop quality filtering
src/cluster_fish.py                Visual clustering
src/config.py                      Project paths and dashboard configuration
src/metrics_reader.py              Reads latest training metrics
src/ui_helpers.py                  UI helper functions
src/build_exe.bat                  Builds the dashboard executable
src/requirements.txt               Python package requirements
```

---

# Pipeline Configuration

Most pipeline settings are near the top of:

```text
src/run_one_video.py
```

Important settings include:

```python
EXTRACT_EVERY_N_FRAMES = 1
SELECT_MAX_FRAMES = 1200
SELECT_MIN_SHARPNESS = 5.0
SELECT_MIN_SCENE_DELTA = 0.02
SELECT_BACKFILL = True
SELECT_MIN_FRAME_GAP = 2

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
CLUSTER_MIN_CLUSTER_SIZE = 8
CLUSTER_MIN_SAMPLES = 5
```

---

# Troubleshooting

## Dashboard does not open

Make sure you are opening:

```text
src/dist/FishAIDashboard/FishAIDashboard.exe
```

Do not open anything from:

```text
src/build/
```

Also make sure `FishAIDashboard.exe` is still beside the `_internal` folder.

---

## Error: missing `ui_helpers`

Rebuild the dashboard:

```bash
cd src
build_exe.bat
```

The build script includes `ui_helpers` as a hidden import.

---

## Error: no module named `cv2`

Install OpenCV:

```bash
pip install opencv-python
```

Or reinstall all requirements:

```bash
pip install -r src\requirements.txt
```

---

## Error: PowerShell security / activate.ps1 blocked

Run this in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Or use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

---

## Error: `best.pt` is 1KB

This means Git LFS has not downloaded the real model.

Run:

```bash
git lfs pull
git lfs checkout
```

If the file is still 1KB, delete the project folder and clone again using:

```bash
git lfs install
git clone https://github.com/DavidWilson2000/COMP3000.git
cd COMP3000
git lfs pull
```

---

## Pipeline says no video found

Place a video in:

```text
data/raw_video/
```

Then run the pipeline again.

---

## Pipeline runs but no crops are produced

Check:

1. `best.pt` exists and is the real model file.
2. The input video contains visible fish.
3. The selected frames folder is not empty.
4. The confidence threshold is not too high.

Try lowering:

```python
CROP_CONF = 0.20
```

---

## Too many false detections

Increase:

```python
CROP_CONF = 0.40
```

Or make filtering stricter:

```python
FILTER_MIN_LONG_SIDE = 128
FILTER_MIN_SHARPNESS = 30.0
FILTER_MIN_CONTRAST = 15.0
```

---

## Most crops go into `unknown_cluster`

Try looser clustering:

```python
CLUSTER_MIN_CLUSTER_SIZE = 5
CLUSTER_MIN_SAMPLES = 2
```

---

# Processed Video Archive

After a video is processed, it is moved to:

```text
data/raw_video/_processed/
```

This prevents the same video from being processed repeatedly.

This is controlled by:

```python
ARCHIVE_PROCESSED_VIDEO = True
```

To stop videos being moved after processing:

```python
ARCHIVE_PROCESSED_VIDEO = False
```

---

# GitHub Notes

Generated outputs such as frames, crops, video runs, and temporary build folders are ignored by Git.

The repository should include:

```text
src/
src/build_exe.bat
src/requirements.txt
data/raw_video/sample_video.mp4
runs/detect/fish_loop/weights/best.pt
datasets/fish_dataset/
README.md
.gitignore
.gitattributes
```

The repository should not include:

```text
runs/video_runs/<full generated runs>
src/build/
src/dist/
Documents/Final Video.mp4
large generated frame/crop folders
```

---

# Summary

To run the project on a new PC:

```bash
git lfs install
git clone https://github.com/DavidWilson2000/COMP3000.git
cd COMP3000
git lfs pull

python -m venv .venv
.venv\Scripts\activate.bat
pip install -r src\requirements.txt

cd src
build_exe.bat
```

Then open:

```text
src/dist/FishAIDashboard/FishAIDashboard.exe
```

To run the pipeline manually instead:

```bash
python src\run_one_video.py
```
