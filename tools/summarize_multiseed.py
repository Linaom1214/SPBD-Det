from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize metrics.json files as mean±std.")
    parser.add_argument("paths", nargs="+", help="metrics.json files or directories containing metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            path = path / "metrics.json"
        with open(path, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    preferred = [
        "mIoU",
        "PixelAcc",
        "threshold_IoU",
        "threshold_nIoU",
        "threshold_F1",
        "threshold_Precision",
        "threshold_Recall",
        "threshold_PixelAcc",
        "loss",
    ]
    keys = [key for key in preferred if any(key in r for r in records)]
    keys.extend(sorted(key for r in records for key in r.keys() if key not in keys and isinstance(r[key], (int, float))))
    for key in keys:
        values = np.array([r[key] for r in records if key in r], dtype=float)
        if values.size:
            print(f"{key}: {values.mean():.6f} ± {values.std(ddof=0):.6f}")


if __name__ == "__main__":
    main()
