from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from sklearn.preprocessing import normalize
import hdbscan


try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# -----------------------------
# Image pre-processing helpers
# -----------------------------
def gray_world_white_balance(np_rgb: np.ndarray) -> np.ndarray:
    """Simple underwater-friendly normalization: equalize channel means (Gray-World WB)."""
    img = np_rgb.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = means.mean()
    scale = gray / means
    img = img * scale
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def hsv_histogram(np_rgb: np.ndarray, h_bins=16, s_bins=8, v_bins=8) -> np.ndarray:
    """Return a normalized HSV histogram feature vector."""
    import cv2
    bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [h_bins, s_bins, v_bins],
        [0, 180, 0, 256, 0, 256]
    )
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def safe_clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_image_paths(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]


# -----------------------------
# Embedding model
# -----------------------------
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
    emb = backbone(x).squeeze(0).cpu().numpy()
    return emb.astype(np.float32)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Cluster fish crops into species-like groups.")
    parser.add_argument("--source", required=True, help="Folder of good fish crops")
    parser.add_argument("--out", required=True, help="Output folder for clustered crops")

    # Quality / speed controls
    parser.add_argument("--max_images", type=int, default=0, help="0 = no limit")

    # Feature controls
    parser.add_argument("--use_color_hist", action="store_true", help="Add HSV histogram features (recommended)")
    parser.add_argument("--use_size_feats", action="store_true", help="Add aspect ratio + relative size features")
    parser.add_argument("--white_balance", action="store_true", help="Apply gray-world white balance before embedding")

    # UMAP controls (recommended)
    parser.add_argument("--use_umap", action="store_true", help="Use UMAP before HDBSCAN (recommended)")
    parser.add_argument("--umap_dim", type=int, default=32, help="UMAP output dims (10-50 typical)")
    parser.add_argument("--umap_neighbors", type=int, default=20, help="UMAP neighbors (10-50 typical)")
    parser.add_argument("--umap_min_dist", type=float, default=0.05, help="UMAP min_dist (0.0-0.5)")

    # HDBSCAN controls
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

    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, transform = build_embedder(device)

    embs = []
    extras = []
    kept_paths = []

    # Collect size stats for "relative size" feature
    sizes = []
    for p in img_paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
                sizes.append((w, h))
        except Exception:
            pass

    if sizes:
        areas = np.array([w * h for w, h in sizes], dtype=np.float32)
        area_med = float(np.median(areas))
    else:
        area_med = 1.0

    for p in img_paths:
        try:
            pil = Image.open(p).convert("RGB")
            np_rgb = np.array(pil)

            if args.white_balance:
                np_rgb = gray_world_white_balance(np_rgb)
                pil = Image.fromarray(np_rgb)

            emb = embed_image(backbone, transform, pil, device)
            embs.append(emb)
            kept_paths.append(p)

            feat_parts = []

            if args.use_color_hist:
                feat_parts.append(hsv_histogram(np_rgb))

            if args.use_size_feats:
                w, h = pil.size
                ar = w / (h + 1e-6)                   # aspect ratio
                rel_area = (w * h) / (area_med + 1e-6) # relative size to median
                feat_parts.append(np.array([ar, rel_area], dtype=np.float32))

            if feat_parts:
                extras.append(np.concatenate(feat_parts).astype(np.float32))
            else:
                extras.append(None)

        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    if not embs:
        raise SystemExit("No embeddings could be computed.")

    X = np.vstack(embs).astype(np.float32)

    # Combine embedding + extra features
    if any(x is not None for x in extras):
        # Replace None with zeros of correct size
        first = next(x for x in extras if x is not None)
        extra_dim = first.shape[0]
        extra_mat = []
        for x in extras:
            if x is None:
                extra_mat.append(np.zeros(extra_dim, dtype=np.float32))
            else:
                extra_mat.append(x)
        extra_mat = np.vstack(extra_mat)

        # Normalize extra features separately then concat
        extra_mat = normalize(extra_mat)

        X = np.concatenate([normalize(X), extra_mat], axis=1)
    else:
        X = normalize(X)

    # Optional UMAP
    if args.use_umap:
        if not UMAP_AVAILABLE:
            print("UMAP requested but umap-learn not installed. Install with: pip install umap-learn")
        else:
            reducer = umap.UMAP(
                n_neighbors=args.umap_neighbors,
                n_components=args.umap_dim,
                min_dist=args.umap_min_dist,
                metric="cosine",
                random_state=0,
            )
            X = reducer.fit_transform(X).astype(np.float32)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",  # after normalize/umap euclidean works well
    )
    labels = clusterer.fit_predict(X)  # -1 = noise

    # Map labels to stable species_001, species_002 by cluster size
    unique = [l for l in sorted(set(labels)) if l != -1]
    cluster_sizes = {l: int(np.sum(labels == l)) for l in unique}
    sorted_clusters = sorted(unique, key=lambda l: cluster_sizes[l], reverse=True)

    label_to_name = {-1: "unknown_cluster"}
    for idx, lab in enumerate(sorted_clusters, start=1):
        label_to_name[lab] = f"species_{idx:03d}"

    # Clear out_dir and write
    safe_clear_dir(out_dir)

    for p, lab in zip(kept_paths, labels):
        group = label_to_name.get(lab, "unknown_cluster")
        dst = out_dir / group
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst / p.name)

    # Write summary CSV
    summary_path = out_dir / "clusters_summary.csv"
    rows = []
    for lab, name in label_to_name.items():
        if lab == -1:
            count = int(np.sum(labels == -1))
        else:
            count = cluster_sizes.get(lab, 0)
        rows.append((name, count))

    rows.sort(key=lambda x: x[1], reverse=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("cluster_name,count\n")
        for name, count in rows:
            f.write(f"{name},{count}\n")

    n_clusters = len(sorted_clusters)
    n_noise = int(np.sum(labels == -1))

    print(f"Found {len(kept_paths)} crops")
    print(f"Clusters: {n_clusters} | Noise: {n_noise}")
    print(f"Output: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
