import argparse
import random
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max", type=int, default=300)
    args = parser.parse_args()

    source_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    frames = [p for p in source_dir.rglob("*") if p.suffix.lower() in exts]

    print(f"Found {len(frames)} total frames in {source_dir}")

    if not frames:
        raise RuntimeError("No frames found to select from.")

    selected = random.sample(frames, min(args.max, len(frames)))

    for frame in selected:
        shutil.copy(frame, out_dir / frame.name)

    print(f"Copied {len(selected)} frames to {out_dir}")

if __name__ == "__main__":
    main()
