from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct the historical IRReversal/ch3 mulframe split from saved ABC prediction names."
    )
    parser.add_argument("--mulframe-root", required=True, help="Root with images/<sequence>/*.png and masks/<sequence>/*.png")
    parser.add_argument("--pred-dir", required=True, help="ABC show directory whose prediction filenames define the test split")
    parser.add_argument("--out-dir", required=True, help="Directory where train_mydataset.txt and test_mydataset.txt are written")
    return parser.parse_args()


def main():
    args = parse_args()
    mulframe_root = Path(args.mulframe_root)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    images_root = mulframe_root / "images"
    masks_root = mulframe_root / "masks"
    pred_names = {p.stem for p in pred_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
    train_names = []
    test_names = []
    missing_masks = []
    for image_path in sorted(images_root.rglob("*.png")):
        name = f"{image_path.parent.name}_{image_path.stem}"
        mask_path = masks_root / image_path.parent.name / image_path.name
        if not mask_path.is_file():
            missing_masks.append(str(mask_path))
            continue
        if name in pred_names:
            test_names.append(name)
        else:
            train_names.append(name)
    if missing_masks:
        raise FileNotFoundError(f"Missing {len(missing_masks)} masks, first: {missing_masks[0]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_mydataset.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (out_dir / "test_mydataset.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")
    print(f"train: {len(train_names)}")
    print(f"test: {len(test_names)}")


if __name__ == "__main__":
    main()
