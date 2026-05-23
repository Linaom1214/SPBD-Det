from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved binary predictions with ABC paper metrics.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--mask-root", required=True, help="Mask root, flat or nested by sequence")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--threshold", type=int, default=0)
    return parser.parse_args()


def resolve_mask(mask_root: Path, name: str) -> Path:
    candidates = [mask_root / f"{name}.png", mask_root / name]
    if "_" in name:
        prefix, frame = name.rsplit("_", 1)
        if frame.isdigit():
            candidates.append(mask_root / prefix / f"{frame}.png")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing mask for {name}")


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    mask_root = Path(args.mask_root)
    names = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    total_correct = 0.0
    total_label = 0.0
    total_inter = 0
    total_union = 0
    pd = 0
    targets = 0
    dismatch_pixel = 0
    all_pixel = 0
    for name in tqdm(names, desc="eval-preds"):
        pred_path = pred_dir / f"{name}.png"
        mask_path = resolve_mask(mask_root, name)
        pred = (np.array(Image.open(pred_path).convert("L")) > args.threshold).astype("int64")
        label = (np.array(Image.open(mask_path).convert("L")) > 127).astype("int64")
        total_label += (label > 0).sum()
        total_correct += ((pred == label) & (label > 0)).sum()
        intersection = pred * (pred == label)
        area_inter, _ = np.histogram(intersection, bins=1, range=(1, 1))
        area_pred, _ = np.histogram(pred, bins=1, range=(1, 1))
        area_label, _ = np.histogram(label, bins=1, range=(1, 1))
        area_union = area_pred + area_label - area_inter
        total_inter += int(area_inter[0])
        total_union += int(area_union[0])

        pred_regions = measure.regionprops(measure.label(pred, connectivity=2))
        label_regions = measure.regionprops(measure.label(label, connectivity=2))
        targets += len(label_regions)
        image_area_total = [np.array(region.area) for region in pred_regions]
        image_area_match = []
        distance_match = []
        for label_region in label_regions:
            centroid_label = np.array(list(label_region.centroid))
            for idx, pred_region in enumerate(pred_regions):
                centroid_pred = np.array(list(pred_region.centroid))
                distance = np.linalg.norm(centroid_pred - centroid_label)
                area_pred_region = np.array(pred_region.area)
                if distance < 3:
                    distance_match.append(distance)
                    image_area_match.append(area_pred_region)
                    del pred_regions[idx]
                    break
        dismatch = [x for x in image_area_total if x not in image_area_match]
        dismatch_pixel += np.sum(dismatch)
        all_pixel += label.shape[0] * label.shape[1]
        pd += len(distance_match)
    eps = np.spacing(1)
    result = {
        "pixAcc": float(total_correct / (total_label + eps)),
        "mIoU": float(total_inter / (total_union + eps)),
        "PD": float(pd / (targets + eps)),
        "FA_raw": float(dismatch_pixel / (all_pixel + eps)),
        "FA_x1e6": float(dismatch_pixel / (all_pixel + eps) * 1e6),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
