from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an existing train/test dataset into train/val/test layout.")
    parser.add_argument("--src-root", required=True, help="Source root with train/images, train/masks, test/images, test/masks")
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio split from source train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    return parser.parse_args()


def find_pairs(root: Path, split: str):
    images_dir = root / split / "images"
    masks_dir = root / split / "masks"
    pairs = []
    for image_path in sorted(p for p in images_dir.iterdir() if p.is_file()):
        candidates = [masks_dir / image_path.name] + list(masks_dir.glob(f"{image_path.stem}.*"))
        mask_path = next((p for p in candidates if p.is_file()), None)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for {image_path}")
        pairs.append((image_path, mask_path))
    return pairs


def link_or_copy(src: Path, dst: Path, copy_file: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_file:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_split(dst_root: Path, split: str, pairs, copy_file: bool):
    for image_path, mask_path in pairs:
        link_or_copy(image_path, dst_root / split / "images" / image_path.name, copy_file)
        link_or_copy(mask_path, dst_root / split / "masks" / image_path.name, copy_file)
    print(f"{split}: {len(pairs)}")


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    train_pairs = find_pairs(src_root, "train")
    test_pairs = find_pairs(src_root, "test")
    random.Random(args.seed).shuffle(train_pairs)
    n_val = int(len(train_pairs) * args.val_ratio)
    val_pairs = train_pairs[:n_val]
    train_pairs = train_pairs[n_val:]
    write_split(dst_root, "train", train_pairs, args.copy)
    write_split(dst_root, "val", val_pairs, args.copy)
    write_split(dst_root, "test", test_pairs, args.copy)


if __name__ == "__main__":
    main()
