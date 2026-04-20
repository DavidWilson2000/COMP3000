from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO
import time

# --------- CONFIG ----------
DATA_DIR = "datasets/species_dataset"
PROJECT = "runs/classify"
RUN_NAME = "species_loop"          # one run folder that keeps growing
CHUNK_EPOCHS = 50                  # train in chunks of 50
IMGSZ = 224
BATCH = 32                         # lower if you get OOM
MODEL_START = "yolov8m-cls.pt"      # first chunk starts here
SLEEP_SECONDS = 5                  # small pause between chunks
# --------------------------

def main():
    run_dir = Path(PROJECT) / RUN_NAME
    weights_dir = run_dir / "weights"
    best_path = weights_dir / "best.pt"

    chunk_idx = 0
    while True:
        chunk_idx += 1

        # pick starting weights: best.pt if it exists, otherwise base model
        model_path = str(best_path) if best_path.exists() else MODEL_START
        print(f"\n=== CHUNK {chunk_idx} | training {CHUNK_EPOCHS} epochs from: {model_path} ===\n")

        model = YOLO(model_path)

        # Train another chunk (saves into the same run folder)
        model.train(
            data=DATA_DIR,
            imgsz=IMGSZ,
            epochs=CHUNK_EPOCHS,
            batch=BATCH,
            project=PROJECT,
            name=RUN_NAME,
            exist_ok=True,
            # resume=False (we're effectively fine-tuning from best each time)
        )

        # Validate and print metrics
        print(f"\n=== CHUNK {chunk_idx} | validating current best ===\n")
        best_model = YOLO(str(best_path) if best_path.exists() else model_path)
        metrics = best_model.val(data=DATA_DIR, imgsz=IMGSZ, batch=BATCH)

        # metrics object prints, but we’ll also give you a clear marker
        print("\n✅ Chunk complete.")
        print(f"Best weights: {best_path}")
        print(f"Run folder:  {run_dir}")
        print("Stopping condition: close the terminal or Ctrl+C.\n")

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
