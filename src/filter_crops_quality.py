import argparse
from pathlib import Path
import shutil
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_long_side", type=int, default=96)
    parser.add_argument("--min_area", type=int, default=96 * 96)
    parser.add_argument("--min_sharpness", type=float, default=20.0)
    args = parser.parse_args()

    in_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0

    for img_path in in_dir.rglob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue

        total += 1
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        long_side = max(h, w)
        area = h * w

        if long_side < args.min_long_side or area < args.min_area:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness < args.min_sharpness:
            continue

        shutil.copy(img_path, out_dir / img_path.name)
        kept += 1

    print(f"Kept {kept}/{total} crops -> {out_dir}")

if __name__ == "__main__":
    main()
