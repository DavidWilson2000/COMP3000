from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from sklearn.preprocessing import normalize
import hdbscan

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def gray_world_white_balance(np_rgb: np.ndarray) -> np.ndarray:
    img = np_rgb.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = means.mean()
    img *= gray / means
    return np.clip(img, 0, 255).astype(np.uint8)


def hsv_histogram(np_rgb: np.ndarray, h_bins: int = 16, s_bins: int = 8, v_bins: int = 8) -> np.ndarray:
    import cv2

    bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def get_image_paths(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]


def safe_clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_embedder(device: str):
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return backbone, transform


@torch.no_grad()
def embed_image(backbone, transform, pil_img: Image.Image, device: str) -> np.ndarray:
    x = transform(pil_img).unsqueeze(0).to(device)
    emb = backbone(x).squeeze(0).cpu().numpy().astype(np.float32)
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster fish crops into species-like groups.")
    parser.add_argument("--source", required=True, help="Folder of good fish crops")
    parser.add_argument("--out", required=True, help="Output folder for clustered crops")
    parser.add_argument("--max_images", type=int, default=0, help="0 = no limit")

    parser.add_argument("--use_color_hist", action="store_true")
    parser.add_argument("--use_size_feats", action="store_true")
    parser.add_argument("--white_balance", action="store_true")

    parser.add_argument("--use_umap", action="store_true")
    parser.add_argument("--umap_dim", type=int, default=32)
    parser.add_argument("--umap_neighbors", type=int, default=20)
    parser.add_argument("--umap_min_dist", type=float, default=0.05)

    parser.add_argument("--min_cluster_size", type=int, default=12)
    parser.add_argument("--min_samples", type=int, default=3)
    args = parser.parse_args()

    crops_dir = Path(args.source)
    out_dir = Path(args.out)

    if not crops_dir.is_dir():
        raise SystemExit(f"Missing crops folder: {crops_dir}")

    img_paths = get_image_paths(crops_dir)
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    if not img_paths:
        print(f"No images found in: {crops_dir} (skipping clustering)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, transform = build_embedder(device)

    embs: list[np.ndarray] = []
    extras: list[np.ndarray | None] = []
    kept_paths: list[Path] = []

    sizes = []
    for p in img_paths:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:
            pass

    area_med = 1.0
    if sizes:
        areas = np.array([w * h for w, h in sizes], dtype=np.float32)
        area_med = float(np.median(areas))

    for p in img_paths:
        try:
            pil = Image.open(p).convert("RGB")
            np_rgb = np.array(pil)

            if args.white_balance:
                np_rgb = gray_world_white_balance(np_rgb)
                pil = Image.fromarray(np_rgb)

            embs.append(embed_image(backbone, transform, pil, device))
            kept_paths.append(p)

            feat_parts = []
            if args.use_color_hist:
                feat_parts.append(hsv_histogram(np_rgb))
            if args.use_size_feats:
                w, h = pil.size
                feat_parts.append(
                    np.array([w / (h + 1e-6), (w * h) / (area_med + 1e-6)], dtype=np.float32)
                )

            extras.append(np.concatenate(feat_parts).astype(np.float32) if feat_parts else None)
        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    if not embs:
        raise SystemExit("No embeddings could be computed.")

    X = np.vstack(embs).astype(np.float32)
    X = normalize(X)

    if any(x is not None for x in extras):
        first = next(x for x in extras if x is not None)
        extra_dim = int(first.shape[0])
        extra_rows = []
        for x in extras:
            extra_rows.append(np.zeros(extra_dim, dtype=np.float32) if x is None else x)
        extra_mat = normalize(np.vstack(extra_rows).astype(np.float32))
        X = np.concatenate([X, extra_mat], axis=1)

    if args.use_umap:
        if not UMAP_AVAILABLE:
            print("UMAP requested but umap-learn is not installed. Continuing without UMAP.")
        elif X.shape[0] < max(10, args.umap_neighbors + 2):
            print(f"Skipping UMAP: only {X.shape[0]} samples for n_neighbors={args.umap_neighbors}.")
        else:
            try:
                reducer = umap.UMAP(
                    n_neighbors=min(args.umap_neighbors, max(2, X.shape[0] - 1)),
                    n_components=min(args.umap_dim, max(2, X.shape[0] - 1)),
                    min_dist=args.umap_min_dist,
                    metric="cosine",
                    random_state=0,
                )
                X = reducer.fit_transform(X).astype(np.float32)
            except Exception as e:
                print(f"UMAP failed ({e}). Continuing without UMAP.")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)

    unique = [l for l in sorted(set(labels)) if l != -1]
    cluster_sizes = {l: int(np.sum(labels == l)) for l in unique}
    sorted_clusters = sorted(unique, key=lambda l: cluster_sizes[l], reverse=True)

    label_to_name = {-1: "unknown_cluster"}
    for idx, lab in enumerate(sorted_clusters, start=1):
        label_to_name[lab] = f"species_{idx:03d}"

    safe_clear_dir(out_dir)

    map_rows = []
    for p, lab in zip(kept_paths, labels):
        group = label_to_name.get(int(lab), "unknown_cluster")
        dst_dir = out_dir / group
        dst_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"{p.parent.name}__{p.name}"
        dst_path = dst_dir / safe_name
        shutil.copy2(p, dst_path)
        map_rows.append([str(p), str(dst_path), int(lab), group])

    summary_path = out_dir / "clusters_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_name", "count"])
        rows = [(label_to_name[-1], int(np.sum(labels == -1)))]
        rows += [(label_to_name[l], cluster_sizes[l]) for l in sorted_clusters]
        for name, count in sorted(rows, key=lambda x: x[1], reverse=True):
            writer.writerow([name, count])

    map_path = out_dir / "cluster_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_path", "copied_path", "cluster_label", "cluster_name"])
        writer.writerows(map_rows)

    print(f"Found {len(kept_paths)} crops")
    print(f"Clusters: {len(sorted_clusters)} | Noise: {int(np.sum(labels == -1))}")
    print(f"Output: {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Map CSV: {map_path}")


if __name__ == "__main__":
    main()