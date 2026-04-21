from pathlib import Path
from ultralytics import YOLO
import time

# ---------------- CONFIG ----------------
FISH_DATA = "datasets/fish_dataset/data.yaml"
SPECIES_DATA = "datasets/species_dataset"

FISH_RUN = "fish_loop"
SPECIES_RUN = "species_loop"

PROJECT_DET = "runs/detect"
PROJECT_CLS = "runs/classify"

DET_MODEL_START = "yolov8m.pt"
CLS_MODEL_START = "yolov8m-cls.pt"

EPOCHS_PER_CHUNK = 50
IMGSZ_DET = 640
IMGSZ_CLS = 224
BATCH = 32
# ---------------------------------------


def train_detector():
    print("\n=== TRAINING FISH DETECTOR ===\n")

    candidates = [
        Path("runs/detect/fish_loop/weights/best.pt"),
        Path("runs/detect/runs/detect/fish_loop/weights/best.pt"),
        Path("runs/detect/fish_detector_v1/weights/best.pt"),
    ]

    best_path = next((p for p in candidates if p.exists()), None)
    model_path = str(best_path) if best_path is not None else DET_MODEL_START

    print(f"Starting detector from: {model_path}")

    model = YOLO(model_path)
    model.train(
        data=FISH_DATA,
        imgsz=IMGSZ_DET,
        epochs=EPOCHS_PER_CHUNK,
        batch=BATCH,
        project=PROJECT_DET,
        name=FISH_RUN,
        exist_ok=True,
    )
    print("\n=== TRAINING FISH DETECTOR ===\n")

    run_dir = Path(PROJECT_DET) / FISH_RUN
    best_path = run_dir / "weights" / "best.pt"

    model_path = str(best_path) if best_path.exists() else DET_MODEL_START

    model = YOLO(model_path)
    model.train(
        data=FISH_DATA,
        imgsz=IMGSZ_DET,
        epochs=EPOCHS_PER_CHUNK,
        batch=BATCH,
        project=PROJECT_DET,
        name=FISH_RUN,
        exist_ok=True,
    )


def train_species():
    print("\n=== TRAINING SPECIES CLASSIFIER ===\n")

    run_dir = Path(PROJECT_CLS) / SPECIES_RUN
    best_path = run_dir / "weights" / "best.pt"

    model_path = str(best_path) if best_path.exists() else CLS_MODEL_START

    model = YOLO(model_path)
    model.train(
        data=SPECIES_DATA,
        imgsz=IMGSZ_CLS,
        epochs=EPOCHS_PER_CHUNK,
        batch=BATCH,
        project=PROJECT_CLS,
        name=SPECIES_RUN,
        exist_ok=True,
    )


def main():
    while True:
        train_detector()
        train_species()
        print("\n=== ONE FULL CYCLE COMPLETE ===\n")
        print("Press Ctrl+C to stop.")
        time.sleep(5)


if __name__ == "__main__":
    main()
