from __future__ import annotations

from pathlib import Path
import shutil
import csv
import argparse

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.models as models

from sklearn.preprocessing import normalize
import hdbscan


# HDBSCAN controls:
MIN_CLUSTER_SIZE = 15   # raise if too many tiny clusters; lower if you have few crops
MIN_SAMPLES = 5         # raise to be stricter (more unknown_cluster)


def get_image_paths(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def build_embedder(device: str):
    # Pretrained ResNet50, remove classifier head => 2048-dim embedding
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return backbone, transform


@torch.no_grad()
def embed_image(backbone, transform, path: Path, device: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    emb = backbone(x).squeeze(0).cpu().numpy()
    return emb


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder containing crops to cluster")
    parser.add_argument("--out", required=True, help="Output folder for clustered groups")
    args = parser.parse_args()

    crops_good_dir = Path(args.source)
    out_dir = Path(args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not crops_good_dir.exists():
        raise SystemExit(f"Missing crops folder: {crops_good_dir}")

    img_paths = get_image_paths(crops_good_dir)
    if not img_paths:
        raise SystemExit(f"No images found in: {crops_good_dir}")

    print(f"Found {len(img_paths)} crops to cluster.")
    clear_dir(out_dir)

    backbone, transform = build_embedder(device)

    embs = []
    kept_paths: list[Path] = []

    for p in img_paths:
        try:
            embs.append(embed_image(backbone, transform, p, device))
            kept_paths.append(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not embs:
        raise SystemExit("No embeddings could be computed.")

    X = np.vstack(embs)
    X = normalize(X)  # cosine-friendly

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
    )
    labels = clusterer.fit_predict(X)  # -1 = noise/outliers

    # Map labels to species_001, species_002...
    unique_labels = sorted(set(labels))
    label_to_name = {}
    species_idx = 1
    for lab in unique_labels:
        if lab == -1:
            label_to_name[lab] = "unknown_cluster"
        else:
            label_to_name[lab] = f"species_{species_idx:03d}"
            species_idx += 1

    # Copy images into cluster folders (safe names to avoid overwrites)
    for p, lab in zip(kept_paths, labels):
        group = label_to_name[lab]
        dst_dir = out_dir / group
        dst_dir.mkdir(parents=True, exist_ok=True)

        safe_name = f"{p.parent.name}__{p.name}"
        shutil.copy(p, dst_dir / safe_name)

    # Save mapping CSV
    map_csv = out_dir / "cluster_map.csv"
    with map_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "cluster_label", "cluster_name"])
        for p, lab in zip(kept_paths, labels):
            w.writerow([str(p), int(lab), label_to_name[lab]])

    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = int(np.sum(labels == -1))

    print(f"Clusters: {n_clusters} | Noise: {n_noise}")
    print(f"Output: {out_dir}")
    print(f"Map CSV: {map_csv}")


if __name__ == "__main__":
    main()
