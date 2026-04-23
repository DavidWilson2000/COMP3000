from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    run_dir: Path | None = None
    results_csv: Path | None = None
    results_png: Path | None = None
    confusion_matrix_png: Path | None = None
    best_weights: Path | None = None
    last_weights: Path | None = None
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    epochs: int | None = None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def find_latest_results_csv(detect_runs_dir: Path) -> Path | None:
    candidates = [p for p in detect_runs_dir.rglob("results.csv") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_results_csv(results_csv: Path) -> dict[str, float | int | None]:
    rows: list[dict[str, str]] = []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    if not rows:
        return {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
            "epochs": None,
        }

    last = rows[-1]

    epoch = None
    for key in ("epoch", "                 epoch", "Epoch"):
        if key in last:
            try:
                epoch = int(float(last[key])) + 1
                break
            except Exception:
                pass

    precision = None
    recall = None
    map50 = None
    map50_95 = None

    # Ultralytics can vary column names slightly by version.
    key_map = {
        "precision": ["metrics/precision(B)", "metrics/precision"],
        "recall": ["metrics/recall(B)", "metrics/recall"],
        "map50": ["metrics/mAP50(B)", "metrics/mAP50"],
        "map50_95": ["metrics/mAP50-95(B)", "metrics/mAP50-95"],
    }

    for key in key_map["precision"]:
        if key in last:
            precision = _safe_float(last[key])
            break
    for key in key_map["recall"]:
        if key in last:
            recall = _safe_float(last[key])
            break
    for key in key_map["map50"]:
        if key in last:
            map50 = _safe_float(last[key])
            break
    for key in key_map["map50_95"]:
        if key in last:
            map50_95 = _safe_float(last[key])
            break

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "epochs": epoch,
    }


def load_latest_run_summary(detect_runs_dir: Path) -> RunSummary:
    results_csv = find_latest_results_csv(detect_runs_dir)
    if results_csv is None:
        return RunSummary()

    run_dir = results_csv.parent
    parsed = parse_results_csv(results_csv)

    summary = RunSummary(
        run_dir=run_dir,
        results_csv=results_csv,
        results_png=run_dir / "results.png",
        confusion_matrix_png=run_dir / "confusion_matrix.png",
        best_weights=run_dir / "weights" / "best.pt",
        last_weights=run_dir / "weights" / "last.pt",
        precision=parsed["precision"],
        recall=parsed["recall"],
        map50=parsed["map50"],
        map50_95=parsed["map50_95"],
        epochs=parsed["epochs"],
    )
    return summary
