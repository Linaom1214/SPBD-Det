from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY_KEYS = [
    "mIoU",
    "PixelAcc",
    "paper_pixAcc",
    "paper_mIoU",
    "paper_PD",
    "paper_FA_raw",
    "paper_FA_x1e6",
    "threshold_IoU",
    "threshold_nIoU",
    "threshold_F1",
    "threshold_Precision",
    "threshold_Recall",
    "threshold_PixelAcc",
    "loss",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ablation metrics as a markdown table.")
    parser.add_argument("paths", nargs="+", help="metrics.json files or directories containing metrics.json")
    return parser.parse_args()


def resolve_metrics_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_dir():
        path = path / "metrics.json"
    return path


def main():
    args = parse_args()
    rows = []
    keys = []
    for raw in args.paths:
        path = resolve_metrics_path(raw)
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        name = path.parent.parent.name if path.parent.name.startswith("eval_") else path.parent.name
        rows.append((name, metrics))
        for key in DISPLAY_KEYS:
            if key in metrics and key not in keys:
                keys.append(key)

    print("| Experiment | " + " | ".join(keys) + " |")
    print("| --- | " + " | ".join(["---:" for _ in keys]) + " |")
    for name, metrics in sorted(rows):
        values = []
        for key in keys:
            value = metrics.get(key)
            values.append("" if value is None else f"{float(value):.6f}")
        print("| " + name + " | " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
