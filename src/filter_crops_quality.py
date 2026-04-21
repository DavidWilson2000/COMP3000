from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class CropInfo:
    path: Path
    width: int
    height: int
    area: int
    aspect_ratio: float
    sharpness: float
    contrast: float
    dhash: int


def compute_dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]

    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def analyze_crop(path: Path) -> CropInfo | None:
    img = cv2.imread(str(path))
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return CropInfo(
        path=path,
        width=w,
        height=h,
        area=w * h,
        aspect_ratio=w / (h + 1e-6),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        contrast=float(gray.std()),
        dhash=compute_dhash(gray),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter fish crops by quality and remove near-duplicates.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_long_side", type=int, default=96)
    parser.add_argument("--min_area", type=int, default=96 * 96)
    parser.add_argument("--min_sharpness", type=float, default=20.0)
    parser.add_argument("--min_contrast", type=float, default=12.0)
    parser.add_argument("--min_aspect_ratio", type=float, default=0.20)
    parser.add_argument("--max_aspect_ratio", type=float, default=5.00)
    parser.add_argument(
        "--dedupe_hamming",
        type=int,
        default=4,
        help="Near-duplicate threshold for dHash. Lower = stricter dedupe.",
    )
    parser.add_argument(
        "--report_csv",
        default="",
        help="Optional CSV path for per-crop keep/reject reasons. Defaults to <out>/filter_report.csv",
    )
    args = parser.parse_args()

    in_dir = Path(args.source)
    out_dir = Path(args.out)

    if not in_dir.is_dir():
        raise RuntimeError(f"Source folder not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    crop_paths = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"Found {len(crop_paths)} crop candidates in {in_dir}")

    infos: list[CropInfo] = []
    for path in crop_paths:
        info = analyze_crop(path)
        if info is not None:
            infos.append(info)

    # Sort so the sharpest version of near-duplicates is kept first.
    infos.sort(key=lambda x: (x.sharpness, x.contrast, x.area), reverse=True)

    seen_hashes: list[int] = []
    kept = 0
    rejected = 0
    report_rows: list[list[str]] = []

    for info in infos:
        long_side = max(info.width, info.height)
        reason = ""

        if long_side < args.min_long_side:
            reason = "too_small_long_side"
        elif info.area < args.min_area:
            reason = "too_small_area"
        elif info.sharpness < args.min_sharpness:
            reason = "too_blurry"
        elif info.contrast < args.min_contrast:
            reason = "too_low_contrast"
        elif info.aspect_ratio < args.min_aspect_ratio or info.aspect_ratio > args.max_aspect_ratio:
            reason = "bad_aspect_ratio"
        elif any(hamming_distance(info.dhash, prev) <= args.dedupe_hamming for prev in seen_hashes):
            reason = "near_duplicate"

        if reason:
            rejected += 1
            report_rows.append([
                str(info.path),
                "reject",
                reason,
                str(info.width),
                str(info.height),
                str(info.area),
                f"{info.aspect_ratio:.4f}",
                f"{info.sharpness:.4f}",
                f"{info.contrast:.4f}",
            ])
            continue

        shutil.copy2(info.path, out_dir / info.path.name)
        seen_hashes.append(info.dhash)
        kept += 1
        report_rows.append([
            str(info.path),
            "keep",
            "passed",
            str(info.width),
            str(info.height),
            str(info.area),
            f"{info.aspect_ratio:.4f}",
            f"{info.sharpness:.4f}",
            f"{info.contrast:.4f}",
        ])

    report_path = Path(args.report_csv) if args.report_csv else (out_dir / "filter_report.csv")
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path",
            "decision",
            "reason",
            "width",
            "height",
            "area",
            "aspect_ratio",
            "sharpness",
            "contrast",
        ])
        writer.writerows(report_rows)

    print(f"Kept {kept}/{len(infos)} crops -> {out_dir}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
