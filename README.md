# Smart Aqua Fish Detection Pipeline

## 1. Project Overview

Smart Aqua is a computer vision pipeline designed to process aquarium video footage and prepare useful fish image data for training, testing, and dataset creation.

The system takes a raw aquarium video, extracts frames from it, selects useful frames, detects fish using a YOLO model, crops the detected fish, filters out low-quality crops, and groups similar fish crops into clusters.

This helps reduce the amount of manual work needed when creating a fish dataset. Instead of manually searching through thousands of video frames, the pipeline automatically produces organised images that can be reviewed, labelled, or uploaded to a tool such as Roboflow.

---

## 2. Main Goal

The main goal of this project is to create a repeatable pipeline for generating useful fish image data from raw video footage.

The project is designed to:

- Extract frames from aquarium videos
- Remove unnecessary or repeated frames
- Detect fish in selected frames
- Crop detected fish into separate images
- Remove poor-quality crops
- Group similar fish images together
- Produce organised folders and CSV reports for review

---

## 3. How the Pipeline Works

The full pipeline is controlled by:

```text
src/run_one_video.py
```

This script runs each stage in order.

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

Each processed video creates a new timestamped run folder inside:

```text
runs/video_runs/
```

Example:

```text
runs/video_runs/aquarium_video__2026-04-30_21-38-08/
```

---

## 4. Project Folder Structure

```text
project-root/
│
├── data/
│   └── raw_video/
│       ├── example_video.mp4
│       └── _processed/
│
├── runs/
│   ├── detect/
│   │   └── fish_loop/
│   │       └── weights/
│   │           └── best.pt
│   │
│   └── video_runs/
│       └── video_name__date_time/
│           ├── frames/
│           ├── selected_frames/
│           ├── crops/
│           ├── crops_good/
│           └── clusters/
│
├── src/
│   ├── run_one_video.py
│   ├── extract_frames.py
│   ├── select_frames_for_labeling.py
│   ├── crop_fish_from_frames.py
│   ├── filter_crops_quality.py
│   └── cluster_fish.py
│
└── README.md
```

---

## 5. Main Controller: `run_one_video.py`

The most important file in this project is:

```text
src/run_one_video.py
```

This file controls the full pipeline. It defines the main settings for each stage, finds the input video, creates the output folders, runs each script, and moves the processed video into an archive folder when finished.

Most changes to the pipeline can be made by editing the configuration values near the top of `run_one_video.py`.

---

## 6. Input Videos

Raw videos should be placed inside:

```text
data/raw_video/
```

Supported video formats are:

```text
.mp4
.mov
.avi
.mkv
```

Example:

```text
data/raw_video/fish_tank_video.mp4
```

When the pipeline runs, it automatically selects the next video from this folder.

---

## 7. Detection Model

The fish detection stage uses a YOLO model.

The preferred model path is:

```text
runs/detect/fish_loop/weights/best.pt
```

The model is selected from this list in `run_one_video.py`:

```python
DET_MODEL_CANDIDATES = [
    Path("runs/detect/fish_loop/weights/best.pt"),
    Path("runs/detect/runs/detect/fish_loop/weights/best.pt"),
    Path("runs/detect/fish_detector_v1/weights/best.pt"),
    Path("yolov8m.pt"),
    Path("yolov8n.pt"),
]
```

The script uses the first model file it can find.

A custom trained fish detection model is recommended because the default YOLO models may not reliably detect aquarium fish.

---

# Pipeline Stages

---

## 8. Stage One: Frame Extraction

### Purpose

The frame extraction stage takes the raw video and saves individual image frames from it.

This is handled by:

```text
src/extract_frames.py
```

### Output Folder

Extracted frames are saved into:

```text
runs/video_runs/<run_name>/frames/
```

### Configuration

Frame extraction is configured in `run_one_video.py`:

```python
EXTRACT_EVERY_N_FRAMES = 1
```

### What This Setting Means

| Value | Meaning |
|---:|---|
| `1` | Save every frame |
| `2` | Save every second frame |
| `5` | Save every fifth frame |
| `10` | Save every tenth frame |

### Recommended Setting

For maximum dataset creation, use:

```python
EXTRACT_EVERY_N_FRAMES = 1
```

For quicker testing, use:

```python
EXTRACT_EVERY_N_FRAMES = 5
```

---

## 9. Stage Two: Frame Selection

### Purpose

The frame selection stage chooses useful frames from the extracted frames.

Aquarium videos often contain thousands of frames that look almost identical. This stage reduces duplication by keeping sharper and more useful frames.

This is handled by:

```text
src/select_frames_for_labeling.py
```

### Output Folder

Selected frames are saved into:

```text
runs/video_runs/<run_name>/selected_frames/
```

### Configuration

Frame selection is configured in `run_one_video.py`:

```python
SELECT_MAX_FRAMES = 1200
SELECT_MIN_SHARPNESS = 5.0
SELECT_MIN_SCENE_DELTA = 0.02
SELECT_BACKFILL = True
SELECT_MIN_FRAME_GAP = 2
```

### Settings Explained

| Setting | Current Value | What It Does |
|---|---:|---|
| `SELECT_MAX_FRAMES` | `1200` | Maximum number of frames to keep. |
| `SELECT_MIN_SHARPNESS` | `5.0` | Removes very blurry frames. Lower values keep more frames. |
| `SELECT_MIN_SCENE_DELTA` | `0.02` | Controls how visually different frames must be. Lower values keep more similar frames. |
| `SELECT_BACKFILL` | `True` | Fills the selected folder with extra usable frames if too few are selected. |
| `SELECT_MIN_FRAME_GAP` | `2` | Minimum gap between selected frame numbers. Lower values allow frames closer together. |

### Important Note

If the project extracts thousands of frames but only selects a very small number, the frame selection settings are probably too strict.

For example, if the pipeline extracts:

```text
3306 frames
```

but only selects:

```text
7 selected frames
```

then the selector is likely rejecting frames because they are too visually similar.

### Recommended Frame Selection Settings

For a good balance between useful data and reduced duplication, use:

```python
SELECT_MAX_FRAMES = 1200
SELECT_MIN_SHARPNESS = 5.0
SELECT_MIN_SCENE_DELTA = 0.02
SELECT_BACKFILL = True
SELECT_MIN_FRAME_GAP = 2
```

### Maximum Frame Retention

To keep almost every usable frame, use:

```python
SELECT_MAX_FRAMES = 3306
SELECT_MIN_SHARPNESS = 0.0
SELECT_MIN_SCENE_DELTA = 0.0
SELECT_BACKFILL = True
SELECT_MIN_FRAME_GAP = 1
```

This is useful when the video has many similar frames but the fish are still visible and useful for training.

---

## 10. Stage Three: Fish Detection and Cropping

### Purpose

The cropping stage uses the YOLO detection model to find fish in the selected frames. Each detected fish is cropped and saved as a separate image.

This is handled by:

```text
src/crop_fish_from_frames.py
```

### Output Folder

Fish crops are saved into:

```text
runs/video_runs/<run_name>/crops/
```

A CSV file is also created:

```text
runs/video_runs/<run_name>/crops/crops.csv
```

This CSV stores metadata about each crop, including the source frame, bounding box coordinates, detection confidence, crop width, and crop height.

### Configuration

Fish detection and cropping are configured in `run_one_video.py`:

```python
CROP_CONF = 0.30
CROP_IMGSZ = 640
CROP_PAD = 0.10
CROP_MIN_SIZE = 40
CROP_MAX_IMAGES = 4000
```

### Settings Explained

| Setting | Current Value | What It Does |
|---|---:|---|
| `CROP_CONF` | `0.30` | Minimum confidence needed to accept a YOLO fish detection. |
| `CROP_IMGSZ` | `640` | Image size used during YOLO inference. |
| `CROP_PAD` | `0.10` | Adds padding around each detected fish before cropping. |
| `CROP_MIN_SIZE` | `40` | Rejects crops smaller than 40 pixels on the shortest side. |
| `CROP_MAX_IMAGES` | `4000` | Maximum number of selected frames to process. |

### If Fish Are Being Missed

Lower the confidence threshold:

```python
CROP_CONF = 0.20
```

### If Too Many False Detections Are Included

Increase the confidence threshold:

```python
CROP_CONF = 0.40
```

### If Crops Are Too Tight

Increase the padding:

```python
CROP_PAD = 0.15
```

### If Crops Contain Too Much Background

Reduce the padding:

```python
CROP_PAD = 0.05
```

---

## 11. Stage Four: Crop Quality Filtering

### Purpose

The crop filtering stage removes poor-quality fish crops.

It can reject crops that are:

- Too small
- Too blurry
- Too low contrast
- The wrong shape
- Near duplicates of another crop

This is handled by:

```text
src/filter_crops_quality.py
```

### Output Folder

Filtered crops are saved into:

```text
runs/video_runs/<run_name>/crops_good/
```

A report is also created:

```text
runs/video_runs/<run_name>/crops_good/filter_report.csv
```

This report shows which crops were kept or rejected and the reason for each decision.

### Configuration

Crop filtering is configured in `run_one_video.py`:

```python
FILTER_MIN_LONG_SIDE = 96
FILTER_MIN_AREA = 96 * 96
FILTER_MIN_SHARPNESS = 20.0
FILTER_MIN_CONTRAST = 12.0
FILTER_MIN_ASPECT_RATIO = 0.30
FILTER_MAX_ASPECT_RATIO = 3.50
FILTER_DEDUPE_HAMMING = 5
```

### Settings Explained

| Setting | Current Value | What It Does |
|---|---:|---|
| `FILTER_MIN_LONG_SIDE` | `96` | Rejects crops where the longest side is smaller than 96 pixels. |
| `FILTER_MIN_AREA` | `96 * 96` | Rejects crops with a total area smaller than 9216 pixels. |
| `FILTER_MIN_SHARPNESS` | `20.0` | Rejects blurry crops. |
| `FILTER_MIN_CONTRAST` | `12.0` | Rejects low-contrast crops. |
| `FILTER_MIN_ASPECT_RATIO` | `0.30` | Rejects crops that are too narrow or tall. |
| `FILTER_MAX_ASPECT_RATIO` | `3.50` | Rejects crops that are too wide or flat. |
| `FILTER_DEDUPE_HAMMING` | `5` | Removes near-duplicate crops using image hashing. |

### If Too Many Crops Are Rejected

Use looser filtering:

```python
FILTER_MIN_LONG_SIDE = 64
FILTER_MIN_AREA = 64 * 64
FILTER_MIN_SHARPNESS = 10.0
FILTER_MIN_CONTRAST = 8.0
FILTER_DEDUPE_HAMMING = 3
```

### If Too Many Bad Crops Are Kept

Use stricter filtering:

```python
FILTER_MIN_LONG_SIDE = 128
FILTER_MIN_AREA = 128 * 128
FILTER_MIN_SHARPNESS = 30.0
FILTER_MIN_CONTRAST = 15.0
FILTER_DEDUPE_HAMMING = 6
```

---

## 12. Stage Five: Fish Crop Clustering

### Purpose

The clustering stage groups similar fish crops together.

This can help with:

- Reviewing similar fish images
- Finding repeated fish appearances
- Organising fish crops into species-like groups
- Preparing data for manual labelling

This is handled by:

```text
src/cluster_fish.py
```

### Output Folder

Clustered crops are saved into:

```text
runs/video_runs/<run_name>/clusters/
```

The clustering stage creates folders such as:

```text
species_001/
species_002/
unknown_cluster/
```

It also creates two CSV files:

```text
clusters_summary.csv
cluster_map.csv
```

### Configuration

Clustering is configured in `run_one_video.py`:

```python
CLUSTER_USE_COLOR_HIST = True
CLUSTER_USE_SIZE_FEATS = True
CLUSTER_WHITE_BALANCE = True
CLUSTER_USE_UMAP = False
CLUSTER_UMAP_DIM = 32
CLUSTER_UMAP_NEIGHBORS = 10
CLUSTER_UMAP_MIN_DIST = 0.05
CLUSTER_MIN_CLUSTER_SIZE = 8
CLUSTER_MIN_SAMPLES = 5
```

### Settings Explained

| Setting | Current Value | What It Does |
|---|---:|---|
| `CLUSTER_USE_COLOR_HIST` | `True` | Uses colour information to help group visually similar fish. |
| `CLUSTER_USE_SIZE_FEATS` | `True` | Uses crop size and aspect ratio as extra clustering features. |
| `CLUSTER_WHITE_BALANCE` | `True` | Applies simple colour correction before clustering. |
| `CLUSTER_USE_UMAP` | `False` | Enables optional dimensionality reduction before clustering. |
| `CLUSTER_UMAP_DIM` | `32` | Number of UMAP dimensions if UMAP is enabled. |
| `CLUSTER_UMAP_NEIGHBORS` | `10` | Controls how local or global the UMAP grouping is. |
| `CLUSTER_UMAP_MIN_DIST` | `0.05` | Controls how tightly UMAP places similar points. |
| `CLUSTER_MIN_CLUSTER_SIZE` | `8` | Minimum number of crops needed to form a cluster. |
| `CLUSTER_MIN_SAMPLES` | `5` | Controls how strict HDBSCAN is when deciding outliers. |

### If Too Many Crops Go Into `unknown_cluster`

Use looser clustering:

```python
CLUSTER_MIN_CLUSTER_SIZE = 5
CLUSTER_MIN_SAMPLES = 2
```

### If Clusters Are Too Messy

Use stricter clustering:

```python
CLUSTER_MIN_CLUSTER_SIZE = 12
CLUSTER_MIN_SAMPLES = 6
```

---

# Running the Project

---

## 13. Step One: Add a Video

Place a video file inside:

```text
data/raw_video/
```

Example:

```text
data/raw_video/aquarium_video.mp4
```

---

## 14. Step Two: Check the Model

Make sure the trained YOLO model exists at:

```text
runs/detect/fish_loop/weights/best.pt
```

---

## 15. Step Three: Run the Pipeline

From the project root folder, run:

```bash
python src/run_one_video.py
```

The script will automatically:

1. Find the next video in `data/raw_video/`
2. Create a new timestamped output folder
3. Extract frames
4. Select useful frames
5. Detect and crop fish
6. Filter crop quality
7. Cluster similar fish crops
8. Move the processed video into `_processed`

---

## 16. Output Files and Folders

Each run creates a new folder inside:

```text
runs/video_runs/
```

Example:

```text
runs/video_runs/aquarium_video__2026-04-30_21-38-08/
```

### Output Folder Breakdown

| Folder or File | Purpose |
|---|---|
| `frames/` | All extracted video frames. |
| `selected_frames/` | Frames selected for fish detection. |
| `crops/` | Raw fish crops created from YOLO detections. |
| `crops/crops.csv` | Metadata for each detected fish crop. |
| `crops_good/` | Crops that passed quality filtering. |
| `crops_good/filter_report.csv` | Report showing why crops were kept or rejected. |
| `clusters/` | Fish crops grouped into species-like folders. |
| `clusters/clusters_summary.csv` | Count of crops in each cluster. |
| `clusters/cluster_map.csv` | Maps each original crop to its cluster output. |

---

# Recommended Configurations

---

## 17. Recommended Full Dataset Configuration

Use this when creating a useful dataset from a full aquarium video:

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
```

This is a good balance between collecting enough data and reducing unnecessary duplicate frames.

---

## 18. Fast Testing Configuration

Use this when you want to quickly test that the pipeline works:

```python
EXTRACT_EVERY_N_FRAMES = 5

SELECT_MAX_FRAMES = 100
SELECT_MIN_SHARPNESS = 5.0
SELECT_MIN_SCENE_DELTA = 0.02
SELECT_BACKFILL = True
SELECT_MIN_FRAME_GAP = 2

CROP_MAX_IMAGES = 100
```

This processes fewer frames and is useful for debugging.

---

## 19. Maximum Data Collection Configuration

Use this when you want to keep as many frames and fish crops as possible:

```python
EXTRACT_EVERY_N_FRAMES = 1

SELECT_MAX_FRAMES = 3306
SELECT_MIN_SHARPNESS = 0.0
SELECT_MIN_SCENE_DELTA = 0.0
SELECT_BACKFILL = True
SELECT_MIN_FRAME_GAP = 1

CROP_CONF = 0.20
CROP_MAX_IMAGES = 5000
```

This may produce more duplicate or low-quality crops, but it is useful when building a large dataset.

---

# Troubleshooting

---

## 20. Problem: Many Frames Extracted but Few Selected

Example problem:

```text
Saved 3306 frames
Selected 7 frames
```

This usually means the frame selection stage is too strict.

Use:

```python
SELECT_MIN_SCENE_DELTA = 0.02
SELECT_BACKFILL = True
SELECT_MIN_FRAME_GAP = 2
```

For maximum retention, use:

```python
SELECT_MIN_SCENE_DELTA = 0.0
SELECT_BACKFILL = True
SELECT_MIN_FRAME_GAP = 1
```

---

## 21. Problem: No Fish Crops Are Saved

Possible causes:

- The selected frames folder is empty
- Too few frames were selected
- The YOLO confidence is too high
- The wrong model is being used
- The detector has not learned the fish properly
- The minimum crop size is too strict

Try:

```python
CROP_CONF = 0.20
CROP_MIN_SIZE = 20
```

Also check that the custom model exists:

```text
runs/detect/fish_loop/weights/best.pt
```

---

## 22. Problem: Too Many False Detections

Increase the detection confidence:

```python
CROP_CONF = 0.40
```

You can also make filtering stricter:

```python
FILTER_MIN_LONG_SIDE = 128
FILTER_MIN_SHARPNESS = 30.0
FILTER_MIN_CONTRAST = 15.0
```

---

## 23. Problem: Too Many Crops Are Rejected

Loosen the filter settings:

```python
FILTER_MIN_LONG_SIDE = 64
FILTER_MIN_AREA = 64 * 64
FILTER_MIN_SHARPNESS = 10.0
FILTER_MIN_CONTRAST = 8.0
```

---

## 24. Problem: Most Crops Go Into `unknown_cluster`

Loosen the clustering settings:

```python
CLUSTER_MIN_CLUSTER_SIZE = 5
CLUSTER_MIN_SAMPLES = 2
```

---

# Processed Video Archive

---

## 25. Video Archiving

By default, once a video has been processed, it is moved into:

```text
data/raw_video/_processed/
```

This prevents the same video from being processed repeatedly.

This is controlled by:

```python
ARCHIVE_PROCESSED_VIDEO = True
```

To stop videos being moved after processing, change it to:

```python
ARCHIVE_PROCESSED_VIDEO = False
```

---

# Dependencies

---

## 26. Required Python Packages

This project uses:

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- Torchvision
- NumPy
- Pillow
- scikit-learn
- HDBSCAN
- UMAP, optional

Install dependencies using:

```bash
pip install ultralytics opencv-python torch torchvision numpy pillow scikit-learn hdbscan umap-learn
```

---

# Summary

---

## 27. Project Summary

Smart Aqua is an automated fish detection and dataset preparation pipeline.

It takes raw aquarium footage and turns it into organised fish image data by:

1. Extracting frames
2. Selecting useful frames
3. Detecting fish
4. Cropping fish images
5. Filtering poor-quality crops
6. Clustering similar fish crops

The main file to configure is:

```text
src/run_one_video.py
```

The most important settings are:

```python
EXTRACT_EVERY_N_FRAMES
SELECT_MAX_FRAMES
SELECT_MIN_SHARPNESS
SELECT_MIN_SCENE_DELTA
SELECT_BACKFILL
SELECT_MIN_FRAME_GAP
CROP_CONF
CROP_MAX_IMAGES
FILTER_MIN_LONG_SIDE
FILTER_MIN_SHARPNESS
CLUSTER_MIN_CLUSTER_SIZE
CLUSTER_MIN_SAMPLES
```

For most full runs, the recommended setup is:

```python
EXTRACT_EVERY_N_FRAMES = 1

SELECT_MAX_FRAMES = 1200
SELECT_MIN_SHARPNESS = 5.0
SELECT_MIN_SCENE_DELTA = 0.02
SELECT_BACKFILL = True
SELECT_MIN_FRAME_GAP = 2

CROP_CONF = 0.30
CROP_MAX_IMAGES = 4000
```

This keeps enough frames for useful dataset creation while still reducing repeated and low-quality data.
