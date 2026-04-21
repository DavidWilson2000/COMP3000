from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class FrameInfo:
    path: Path
    sharpness: float
    brightness: float
    signature: np.ndarray
    frame_idx: int


def iter_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def extract_frame_index(path: Path) -> int:
    m = re.search(r"frame_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def compute_signature(bgr: np.ndarray, size: int = 24) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    sig = small.astype(np.float32).flatten()
    norm = np.linalg.norm(sig)
    if norm > 0:
        sig /= norm
    return sig


def sharpness_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def build_frame_infos(paths: list[Path], min_sharpness: float) -> list[FrameInfo]:
    infos: list[FrameInfo] = []

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue

        sharp = sharpness_score(img)
        if sharp < min_sharpness:
            continue

        infos.append(
            FrameInfo(
                path=path,
                sharpness=sharp,
                brightness=brightness_score(img),
                signature=compute_signature(img),
                frame_idx=extract_frame_index(path),
            )
        )

    return infos


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return 1.0 - sim


def greedy_diverse_select(
    infos: list[FrameInfo],
    max_keep: int,
    min_scene_delta: float,
    min_frame_gap: int,
    backfill: bool,
) -> list[FrameInfo]:
    if not infos:
        return []

    infos_sorted = sorted(
        infos,
        key=lambda x: (x.sharpness, -abs(x.brightness - 110.0)),
        reverse=True,
    )

    selected: list[FrameInfo] = []

    for info in infos_sorted:
        if len(selected) >= max_keep:
            break

        if not selected:
            selected.append(info)
            continue

        too_close_in_time = any(
            s.frame_idx >= 0 and info.frame_idx >= 0 and abs(info.frame_idx - s.frame_idx) < min_frame_gap
            for s in selected
        )
        if too_close_in_time:
            continue

        min_dist = min(signature_distance(info.signature, s.signature) for s in selected)
        if min_dist >= min_scene_delta:
            selected.append(info)

    if backfill and len(selected) < min(max_keep, len(infos_sorted)):
        selected_paths = {x.path for x in selected}
        for info in infos_sorted:
            if len(selected) >= max_keep:
                break
            if info.path in selected_paths:
                continue

            too_close_in_time = any(
                s.frame_idx >= 0 and info.frame_idx >= 0 and abs(info.frame_idx - s.frame_idx) < min_frame_gap
                for s in selected
            )
            if too_close_in_time:
                continue

            selected.append(info)
            selected_paths.add(info.path)

    return sorted(selected, key=lambda x: x.frame_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select sharper, more visually distinct frames.")
    parser.add_argument("--source", required=True, help="Folder containing extracted video frames")
    parser.add_argument("--out", required=True, help="Output folder for selected frames")
    parser.add_argument("--max", type=int, default=300, help="Maximum number of frames to keep")
    parser.add_argument("--min_sharpness", type=float, default=15.0)
    parser.add_argument("--min_scene_delta", type=float, default=0.12)
    parser.add_argument("--min_frame_gap", type=int, default=20, help="Minimum gap in frame numbers between selected frames")
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    out_dir = Path(args.out).resolve()

    if not source_dir.is_dir():
        raise RuntimeError(f"Source directory not found: {source_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = iter_images(source_dir)
    print(f"Found {len(frame_paths)} total frames in {source_dir}")

    if not frame_paths:
        raise RuntimeError("No frames found to select from.")

    infos = build_frame_infos(frame_paths, min_sharpness=args.min_sharpness)
    print(f"Usable after sharpness filter: {len(infos)}")

    if not infos:
        raise RuntimeError("No frames met the sharpness threshold.")

    selected = greedy_diverse_select(
        infos=infos,
        max_keep=max(1, args.max),
        min_scene_delta=max(0.0, args.min_scene_delta),
        min_frame_gap=max(0, args.min_frame_gap),
        backfill=args.backfill,
    )

    for info in selected:
        shutil.copy2(info.path, out_dir / info.path.name)

    print(f"Selected {len(selected)} frames -> {out_dir}")
    if selected:
        avg_sharp = sum(x.sharpness for x in selected) / len(selected)
        print(f"Average sharpness of selected frames: {avg_sharp:.2f}")


if __name__ == "__main__":
    main()