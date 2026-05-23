from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY_KEYS = [
    "params_m",
    "encoder_params",
    "decoder_params",
    "flops_g",
    "latency_ms",
    "fps",
    "peak_memory_mb",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize model_complexity.py JSON files as a markdown table.")
    parser.add_argument("paths", nargs="+", help="complexity JSON files or directories containing complexity.json")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_dir():
        path = path / "complexity.json"
    return path


def fmt(key: str, value):
    if value is None:
        return ""
    if key.endswith("params"):
        return f"{int(value):d}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    args = parse_args()
    rows = []
    for raw in args.paths:
        path = resolve_path(raw)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = path.parent.name if path.name == "complexity.json" else path.stem
        rows.append((name, data))

    print("| Experiment | Deploy | " + " | ".join(DISPLAY_KEYS) + " |")
    print("| --- | ---: | " + " | ".join(["---:" for _ in DISPLAY_KEYS]) + " |")
    for name, data in sorted(rows):
        values = [fmt(key, data.get(key)) for key in DISPLAY_KEYS]
        print("| " + name + " | " + str(data.get("deploy", False)) + " | " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
