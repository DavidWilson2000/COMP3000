from __future__ import annotations

from pathlib import Path
import argparse

from ultralytics import YOLO

# ---------------- CONFIG ----------------
FISH_DATA = "datasets/fish_dataset/data.yaml"
PROJECT_DET = "runs/detect/runs/detect"
FISH_RUN = "fish_loop"

DET_MODEL_START = "yolov8m.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 512
DEFAULT_BATCH = 8
# ---------------------------------------


def train_detector_once(epochs: int, imgsz: int, batch: int) -> Path:
    run_dir = Path(PROJECT_DET) / FISH_RUN
    best_path = run_dir / "weights" / "best.pt"

    model_path = str(best_path) if best_path.exists() else DET_MODEL_START
    print(f"\n=== TRAINING FISH DETECTOR ===")
    print(f"Starting from: {model_path}")
    print(f"Dataset:       {FISH_DATA}")
    print(f"Run folder:    {run_dir}")
    print(f"Epochs:        {epochs}")
    print(f"Image size:    {imgsz}")
    print(f"Batch:         {batch}\n")

    model = YOLO(model_path)
    model.train(
        data=FISH_DATA,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        project=PROJECT_DET,
        name=FISH_RUN,
        exist_ok=True,
    )

    final_best = run_dir / "weights" / "best.pt"
    print("\n✅ Detector training complete.")
    print(f"Best weights: {final_best}")
    return final_best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train only the fish detector used by the clustering pipeline."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs for this training run")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Training image size")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size")
    args = parser.parse_args()

    train_detector_once(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)


if __name__ == "__main__":
    main()
