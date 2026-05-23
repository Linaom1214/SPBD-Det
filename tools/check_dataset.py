from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spdnet.utils.config import load_config, merge_overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Check expected dataset layout for SPD.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cfg-options", nargs="*", default=None)
    return parser.parse_args()


def resolve_indexed_path(base_dir, name, suffixes):
    raw = Path(name)
    candidates = []
    if raw.suffix:
        candidates.append(base_dir / raw)
    else:
        candidates.extend(base_dir / f"{name}.{suffix}" for suffix in suffixes)
    if "_" in name:
        prefix, frame = name.rsplit("_", 1)
        if frame.isdigit():
            candidates.extend(base_dir / prefix / f"{frame}.{suffix}" for suffix in suffixes)
    return next((p for p in candidates if p.is_file()), None)


def count_split(root, split, image_dir, mask_dir, suffixes, split_file=None):
    suffixes = tuple(s.lower().lstrip(".") for s in suffixes)
    if split_file:
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = root / split_path
        if not split_path.is_file():
            return {"split": split, "exists": False, "images": 0, "masks": 0, "missing_masks": None, "split_file": str(split_path)}
        names = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        images = root / image_dir
        masks = root / mask_dir
        missing = [name for name in names if resolve_indexed_path(images, name, suffixes) is None or resolve_indexed_path(masks, name, suffixes) is None]
        return {"split": split, "exists": True, "images": len(names) - len(missing), "masks": len(names) - len(missing), "missing_masks": len(missing), "split_file": str(split_path)}
    split_root = root / split
    images = split_root / image_dir
    masks = split_root / mask_dir
    if not images.is_dir() or not masks.is_dir():
        return {"split": split, "exists": False, "images": 0, "masks": 0, "missing_masks": None}
    image_files = sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower().lstrip(".") in suffixes)
    mask_names = {p.name for p in masks.iterdir() if p.is_file()}
    mask_stems = {p.stem for p in masks.iterdir() if p.is_file()}
    missing = [p.name for p in image_files if p.name not in mask_names and p.stem not in mask_stems]
    return {"split": split, "exists": True, "images": len(image_files), "masks": len(mask_names), "missing_masks": len(missing)}


def main():
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), args.cfg_options)
    root = Path(cfg.data.root)
    splits = [cfg.data.train_dir, cfg.data.val_dir, cfg.data.test_dir]
    print(f"Dataset root: {root.resolve()}")
    for name, split in zip(["train", "val", "test"], splits):
        split_file = getattr(cfg.data, f"{name}_split_file", None)
        print(count_split(root, split, cfg.data.image_dir, cfg.data.mask_dir, cfg.data.suffixes, split_file))


if __name__ == "__main__":
    main()
