from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect fish in frames and save crops.")
    parser.add_argument("--source", required=True, help="Folder containing frames")
    parser.add_argument("--out", required=True, help="Output folder for fish crops")
    parser.add_argument("--model", required=True, help="Path to fish detector weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Detector inference image size")
    parser.add_argument("--max_images", type=int, default=2000, help="Max number of frames to process")
    parser.add_argument("--pad", type=float, default=0.15, help="BBox padding fraction")
    parser.add_argument("--min_crop", type=int, default=32, help="Skip crops smaller than this (pixels)")
    parser.add_argument(
        "--csv_path",
        default="",
        help="Optional CSV path for crop metadata. Defaults to <out>/crops.csv",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    out_dir = Path(args.out)
    model_path = Path(args.model)

    if not source_dir.is_dir():
        raise RuntimeError(f"Source folder not found: {source_dir}")
    if not model_path.is_file():
        raise RuntimeError(f"Model not found: {model_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    images = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    images = images[: max(1, args.max_images)]

    print(f"Found {len(images)} images in {source_dir} (processing up to {args.max_images}).")

    csv_path = Path(args.csv_path) if args.csv_path else (out_dir / "crops.csv")
    rows = []
    crop_count = 0
    frame_count = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        frame_count += 1
        h, w = img.shape[:2]

        results = model.predict(source=img, imgsz=args.imgsz, conf=args.conf, verbose=False)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            continue

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            det_conf = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0

            bw = x2 - x1
            bh = y2 - y1
            x1 = int(max(0, x1 - bw * args.pad))
            y1 = int(max(0, y1 - bh * args.pad))
            x2 = int(min(w, x2 + bw * args.pad))
            y2 = int(min(h, y2 + bh * args.pad))

            if x2 <= x1 or y2 <= y1:
                continue

            crop_w = x2 - x1
            crop_h = y2 - y1
            if min(crop_w, crop_h) < args.min_crop:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            safe_stem = f"{img_path.parent.name}__{img_path.stem}"
            out_path = out_dir / f"{safe_stem}_fish_{i:02d}.jpg"
            cv2.imwrite(str(out_path), crop)
            crop_count += 1

            rows.append([
                str(img_path),
                str(out_path),
                i,
                x1,
                y1,
                x2,
                y2,
                f"{det_conf:.6f}",
                crop_w,
                crop_h,
            ])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_path",
            "crop_path",
            "det_idx",
            "x1",
            "y1",
            "x2",
            "y2",
            "det_conf",
            "crop_width",
            "crop_height",
        ])
        writer.writerows(rows)

    print(f"Processed {frame_count} frames.")
    print(f"Saved {crop_count} fish crops to {out_dir}")
    print(f"Saved crop metadata to: {csv_path}")


if __name__ == "__main__":
    main()
