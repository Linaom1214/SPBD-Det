from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test splits from images/ and masks/ folders.")
    parser.add_argument("--src-root", required=True, help="Source root with images/ and masks/")
    parser.add_argument("--dst-root", required=True, help="Destination root with train/val/test folders")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy_file: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_file:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    args = parse_args()
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train + val + test must equal 1")
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    images_dir = src_root / "images"
    masks_dir = src_root / "masks"
    image_paths = sorted(p for p in images_dir.iterdir() if p.is_file())
    pairs = []
    for image_path in image_paths:
        candidates = [masks_dir / image_path.name] + list(masks_dir.glob(f"{image_path.stem}.*"))
        mask_path = next((p for p in candidates if p.is_file()), None)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        pairs.append((image_path, mask_path))
    random.Random(args.seed).shuffle(pairs)
    n = len(pairs)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    split_pairs = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }
    for split, items in split_pairs.items():
        for image_path, mask_path in items:
            link_or_copy(image_path, dst_root / split / "images" / image_path.name, args.copy)
            link_or_copy(mask_path, dst_root / split / "masks" / image_path.name, args.copy)
        print(f"{split}: {len(items)}")


if __name__ == "__main__":
    main()
