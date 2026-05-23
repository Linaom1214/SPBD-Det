from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.visualize_failure_cases import (  # noqa: E402
    bbox_from_masks,
    draw_box,
    load_binary_mask,
    make_error_map,
    make_overlay,
    resize_binary,
    safe_name,
)


FIELDNAMES = [
    "dataset",
    "category",
    "rank_in_category",
    "case_id",
    "name",
    "iou",
    "precision",
    "recall",
    "tp_pixels",
    "fp_pixels",
    "fn_pixels",
    "gt_pixels",
    "pred_pixels",
    "target_components",
    "pred_components",
    "detected_targets",
    "missed_targets",
    "false_alarm_components",
    "false_alarm_pixels",
    "review_dir",
    "image_path",
    "mask_path",
    "prediction_path",
    "suggested_analysis",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect representative failure cases for manual review.")
    parser.add_argument("--out-dir", default="work_dirs/failure_cases/manual_review")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--ranking", action="append", nargs=2, metavar=("DATASET", "CSV"), required=True)
    parser.add_argument("--pred-root", action="append", nargs=2, metavar=("DATASET", "PRED_ROOT"), required=True)
    return parser.parse_args()


def read_ranking(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def float_value(record: dict[str, str], key: str) -> float:
    return float(record[key])


def int_value(record: dict[str, str], key: str) -> int:
    return int(float(record[key]))


def select_cases(records: list[dict[str, str]], per_category: int) -> dict[str, list[dict[str, str]]]:
    non_empty = [r for r in records if int_value(r, "gt_pixels") > 0 or int_value(r, "pred_pixels") > 0]
    return {
        "worst_iou": sorted(non_empty, key=lambda r: (float_value(r, "iou"), -int_value(r, "fp_pixels") - int_value(r, "fn_pixels")))[:per_category],
        "false_negative": sorted(
            [r for r in non_empty if int_value(r, "fn_pixels") > 0],
            key=lambda r: (int_value(r, "missed_targets"), int_value(r, "fn_pixels"), 1.0 - float_value(r, "recall")),
            reverse=True,
        )[:per_category],
        "false_positive": sorted(
            [r for r in non_empty if int_value(r, "fp_pixels") > 0],
            key=lambda r: (int_value(r, "false_alarm_components"), int_value(r, "fp_pixels"), 1.0 - float_value(r, "precision")),
            reverse=True,
        )[:per_category],
        "mixed": sorted(
            [r for r in non_empty if int_value(r, "fp_pixels") > 0 and int_value(r, "fn_pixels") > 0],
            key=lambda r: ((1.0 - float_value(r, "iou")) * (int_value(r, "fp_pixels") + int_value(r, "fn_pixels"))),
            reverse=True,
        )[:per_category],
    }


def resolve_prediction(pred_root: Path, name: str) -> Path:
    candidates = [pred_root / name, pred_root / safe_name(name), pred_root / f"{name}.png", pred_root / f"{safe_name(name)}.png"]
    if "_" in name:
        prefix, frame = name.rsplit("_", 1)
        if frame.isdigit():
            candidates.append(pred_root / prefix / f"{frame}.png")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve prediction for {name} under {pred_root}")


def load_case_arrays(record: dict[str, str], pred_path: Path):
    pred = load_binary_mask(pred_path, 0)
    gt = load_binary_mask(Path(record["mask_path"]), 127)
    if gt.shape != pred.shape:
        gt = resize_binary(gt, pred.shape)
    image = Image.open(record["image_path"]).convert("RGB").resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
    return np.array(image), gt, pred


def crop_around_failure(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, min_size: int = 96, pad: int = 32):
    box = bbox_from_masks(gt, pred, pad=pad)
    if box is None:
        return image, gt, pred, None
    x0, y0, x1, y1 = box
    h, w = gt.shape
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    size = max(min_size, x1 - x0 + 1, y1 - y0 + 1)
    half = size // 2
    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(x0 + size, w)
    y1 = min(y0 + size, h)
    x0 = max(x1 - size, 0)
    y0 = max(y1 - size, 0)
    crop_box = (x0, y0, x1, y1)
    return image[y0:y1, x0:x1], gt[y0:y1, x0:x1], pred[y0:y1, x0:x1], crop_box


def save_mask(path: Path, mask: np.ndarray):
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def save_review_panel(record: dict[str, str], image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: Path, title_prefix: str):
    box = bbox_from_masks(gt, pred, pad=12)
    panels = [
        draw_box(image, box, (255, 255, 0)),
        np.repeat((gt.astype(np.uint8) * 255)[:, :, None], 3, axis=2),
        np.repeat((pred.astype(np.uint8) * 255)[:, :, None], 3, axis=2),
        draw_box(make_overlay(image, gt, pred), box, (255, 255, 0)),
        draw_box(make_error_map(gt, pred), box, (255, 255, 0)),
    ]
    titles = ["Image", "GT", "Prediction", "Overlay", "Error: green TP / red FP / blue FN"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), dpi=260)
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"{title_prefix} | IoU={float_value(record, 'iou'):.3f}, "
        f"P={float_value(record, 'precision'):.3f}, R={float_value(record, 'recall'):.3f}, "
        f"FP={int_value(record, 'fp_pixels')}, FN={int_value(record, 'fn_pixels')}",
        fontsize=8,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def suggested_analysis(record: dict[str, str], category: str) -> str:
    missed = int_value(record, "missed_targets")
    fa = int_value(record, "false_alarm_components")
    fp = int_value(record, "fp_pixels")
    fn = int_value(record, "fn_pixels")
    if category == "false_negative" or (fn > fp and missed > 0):
        return "候选分析：目标极小或局部对比度弱，预测区域偏小/缺失，表现为漏检或目标边界召回不足。"
    if category == "false_positive" or fp > fn:
        return "候选分析：背景中存在点状高亮或结构化杂波，被模型误判为小目标，表现为额外连通域误检。"
    if category == "mixed":
        return "候选分析：目标响应与背景杂波同时存在，模型对真实目标定位不足，同时在邻近背景产生误检。"
    if missed > 0 and fa > 0:
        return "候选分析：真实目标完全漏检，同时背景中出现多个相似点状响应。"
    return "候选分析：单图 IoU 较低，需要人工检查目标标注、预测偏移和背景杂波是否导致评分下降。"


def write_readme(out_dir: Path):
    text = """# Failure-case manual review set

Each case directory contains:

- `panel_full.png`: full-image five-column visualization.
- `panel_crop.png`: zoomed crop around GT/prediction/error pixels.
- `image.png`: resized source image used for review.
- `gt.png`: binary ground-truth mask.
- `pred.png`: binary prediction mask.
- `overlay.png`: green/red overlay on the image. Green means GT/pred overlap, red means predicted foreground.
- `error.png`: error map. Green = TP, red = FP, blue = FN.
- `metadata.json`: metrics, source paths, and a candidate failure-mode explanation.

Use `review_index.csv` to mark whether the automatically assigned category and suggested explanation are correct.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_paths = {dataset: Path(path) for dataset, path in args.ranking}
    pred_roots = {dataset: Path(path) for dataset, path in args.pred_root}
    review_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    for dataset, ranking_path in ranking_paths.items():
        records = read_ranking(ranking_path)
        categories = select_cases(records, args.per_category)
        pred_root = pred_roots[dataset]
        for category, selected in categories.items():
            for rank, record in enumerate(selected, start=1):
                key = (dataset, category, record["name"])
                if key in seen:
                    continue
                seen.add(key)
                case_id = f"{dataset}_{category}_{rank:02d}_{safe_name(record['name'])}"
                case_dir = out_dir / dataset / category / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                pred_path = resolve_prediction(pred_root, record["name"])
                image, gt, pred = load_case_arrays(record, pred_path)
                crop_image, crop_gt, crop_pred, crop_box = crop_around_failure(image, gt, pred)

                Image.fromarray(image).save(case_dir / "image.png")
                save_mask(case_dir / "gt.png", gt)
                save_mask(case_dir / "pred.png", pred)
                Image.fromarray(make_overlay(image, gt, pred)).save(case_dir / "overlay.png")
                Image.fromarray(make_error_map(gt, pred)).save(case_dir / "error.png")
                Image.fromarray(crop_image).save(case_dir / "crop_image.png")
                save_mask(case_dir / "crop_gt.png", crop_gt)
                save_mask(case_dir / "crop_pred.png", crop_pred)
                Image.fromarray(make_overlay(crop_image, crop_gt, crop_pred)).save(case_dir / "crop_overlay.png")
                Image.fromarray(make_error_map(crop_gt, crop_pred)).save(case_dir / "crop_error.png")
                save_review_panel(record, image, gt, pred, case_dir / "panel_full.png", f"{dataset} | {category} | {record['name']}")
                save_review_panel(record, crop_image, crop_gt, crop_pred, case_dir / "panel_crop.png", f"{dataset} | {category} | crop | {record['name']}")

                shutil.copy2(record["image_path"], case_dir / f"source_image{Path(record['image_path']).suffix}")
                shutil.copy2(record["mask_path"], case_dir / f"source_gt{Path(record['mask_path']).suffix}")
                shutil.copy2(pred_path, case_dir / f"source_pred{pred_path.suffix}")

                analysis = suggested_analysis(record, category)
                metadata = {
                    **record,
                    "dataset": dataset,
                    "category": category,
                    "rank_in_category": rank,
                    "case_id": case_id,
                    "prediction_path": str(pred_path),
                    "review_dir": str(case_dir),
                    "crop_box_xyxy": crop_box,
                    "suggested_analysis": analysis,
                    "manual_review": {
                        "category_correct": "",
                        "analysis_correct": "",
                        "notes": "",
                    },
                }
                (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
                review_rows.append(metadata)

    with open(out_dir / "review_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES + ["category_correct", "analysis_correct", "manual_notes"])
        writer.writeheader()
        for row in review_rows:
            writer.writerow({
                **{field: row.get(field, "") for field in FIELDNAMES},
                "category_correct": "",
                "analysis_correct": "",
                "manual_notes": "",
            })
    write_readme(out_dir)
    print(json.dumps({"out_dir": str(out_dir), "num_cases": len(review_rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
