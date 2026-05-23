from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage import measure
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spdnet.data import build_dataloader  # noqa: E402
from spdnet.models import SPD  # noqa: E402
from spdnet.utils.config import load_config, merge_overrides  # noqa: E402
from spdnet.utils.reproducibility import seed_everything  # noqa: E402


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Find and visualize segmentation failure cases.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--grid-count", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", help="Run model inference from a released SPD config.")
    mode.add_argument("--pred-dir", help="Use an existing directory of saved binary predictions.")

    parser.add_argument("--checkpoint", help="Checkpoint for --config mode.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--cfg-options", nargs="*", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save-predictions", action="store_true", help="Keep checkpoint-mode binary predictions.")

    parser.add_argument("--image-root", help="Image root for --pred-dir mode.")
    parser.add_argument("--mask-root", help="Mask root for --pred-dir mode.")
    parser.add_argument("--split-file", help="Split file for --pred-dir mode.")
    parser.add_argument("--suffixes", nargs="+", default=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    return parser.parse_args()


def safe_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__")


def resolve_indexed_path(base_dir: Path, name: str, suffixes: list[str]) -> Path:
    raw = Path(name)
    candidates: list[Path] = []
    if raw.suffix:
        candidates.append(base_dir / raw)
    else:
        candidates.extend(base_dir / f"{name}.{suffix}" for suffix in suffixes)
    if "_" in name:
        prefix, frame = name.rsplit("_", 1)
        if frame.isdigit():
            candidates.extend(base_dir / prefix / f"{frame}.{suffix}" for suffix in suffixes)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve {name} under {base_dir}")


def load_binary_mask(path: Path, threshold: float | int = 0) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr > threshold


def resize_binary(mask: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(image.resize((w, h), Image.NEAREST)) > 127


def denormalize_image(tensor: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    image = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    image = image * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def component_stats(gt: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    gt_label = measure.label(gt.astype(np.uint8), connectivity=2)
    pred_label = measure.label(pred.astype(np.uint8), connectivity=2)
    gt_regions = measure.regionprops(gt_label)
    pred_regions = measure.regionprops(pred_label)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for gi, gt_region in enumerate(gt_regions):
        gy, gx = gt_region.centroid
        for pi, pred_region in enumerate(pred_regions):
            py, px = pred_region.centroid
            if (gx - px) ** 2 + (gy - py) ** 2 < 9.0:
                matched_gt.add(gi)
                matched_pred.add(pi)
    false_alarm_pixels = 0
    for pi, pred_region in enumerate(pred_regions):
        if pi not in matched_pred:
            false_alarm_pixels += int(pred_region.area)
    return {
        "target_components": len(gt_regions),
        "pred_components": len(pred_regions),
        "detected_targets": len(matched_gt),
        "missed_targets": len(gt_regions) - len(matched_gt),
        "false_alarm_components": len(pred_regions) - len(matched_pred),
        "false_alarm_pixels": false_alarm_pixels,
    }


def compute_record(dataset_name: str, name: str, pred: np.ndarray, gt: np.ndarray, image_path: str = "", mask_path: str = "") -> dict[str, object]:
    if pred.shape != gt.shape:
        gt = resize_binary(gt, pred.shape)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    gt_pixels = int(gt.sum())
    pred_pixels = int(pred.sum())
    union = tp + fp + fn
    iou = float(tp / union) if union else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) else (1.0 if gt_pixels == 0 else 0.0)
    recall = float(tp / (tp + fn)) if (tp + fn) else 1.0
    stats = component_stats(gt, pred)
    return {
        "dataset": dataset_name,
        "name": name,
        "image_path": image_path,
        "mask_path": mask_path,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        **stats,
    }


def run_checkpoint_mode(args, out_dir: Path) -> tuple[list[dict[str, object]], dict[str, tuple[Path, Path, Path]]]:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required with --config")
    cfg = merge_overrides(load_config(args.config), args.cfg_options)
    seed_everything(int(cfg.seed), bool(cfg.deterministic))
    if args.device == "auto":
        device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    else:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    loader = build_dataloader(cfg, args.split, training=False)
    model = SPD(**cfg.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if state_dict and all(k.startswith("decode_head.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("decode_head."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sample_lookup = {sample_name: (image_path, mask_path) for image_path, mask_path, sample_name in loader.dataset.samples}
    records: list[dict[str, object]] = []
    panel_lookup: dict[str, tuple[Path, Path, Path]] = {}
    mean = np.array(cfg.data.normalize_mean, dtype=np.float32)
    std = np.array(cfg.data.normalize_std, dtype=np.float32)
    with torch.no_grad():
        for images, masks, names in tqdm(loader, desc=f"failure-scan-{args.dataset_name}"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > float(args.threshold)).cpu().numpy()
            masks_np = masks.numpy()[:, 0] > 0.5
            for batch_index, name in enumerate(names):
                pred = preds[batch_index]
                gt = masks_np[batch_index]
                pred_path = pred_dir / safe_name(name)
                if pred_path.suffix == "":
                    pred_path = pred_path.with_suffix(".png")
                Image.fromarray(pred.astype(np.uint8) * 255).save(pred_path)
                image_path, mask_path = sample_lookup[name]
                record = compute_record(args.dataset_name, name, pred, gt, str(image_path), str(mask_path))
                records.append(record)
                panel_lookup[name] = (image_path, mask_path, pred_path)
                if not args.save_predictions:
                    # Keep predictions by default because selected panels are made after ranking.
                    pass
    (out_dir / "normalization.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2), encoding="utf-8"
    )
    return records, panel_lookup


def run_saved_prediction_mode(args) -> tuple[list[dict[str, object]], dict[str, tuple[Path, Path, Path]]]:
    image_root = Path(args.image_root)
    mask_root = Path(args.mask_root)
    pred_dir = Path(args.pred_dir)
    split_names = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    records: list[dict[str, object]] = []
    panel_lookup: dict[str, tuple[Path, Path, Path]] = {}
    for name in tqdm(split_names, desc=f"failure-scan-{args.dataset_name}"):
        image_path = resolve_indexed_path(image_root, name, args.suffixes)
        mask_path = resolve_indexed_path(mask_root, name, args.suffixes)
        pred_path = resolve_indexed_path(pred_dir, name, ["png"])
        pred = load_binary_mask(pred_path, 0)
        gt = load_binary_mask(mask_path, 127)
        if pred.shape != gt.shape:
            gt = resize_binary(gt, pred.shape)
        records.append(compute_record(args.dataset_name, name, pred, gt, str(image_path), str(mask_path)))
        panel_lookup[name] = (image_path, mask_path, pred_path)
    return records, panel_lookup


def bbox_from_masks(gt: np.ndarray, pred: np.ndarray, pad: int = 18) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.logical_or(gt, pred))
    if len(xs) == 0:
        return None
    h, w = gt.shape
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, w - 1)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, h - 1)
    return x0, y0, x1, y1


def draw_box(image: np.ndarray, box: tuple[int, int, int, int] | None, color: tuple[int, int, int]) -> np.ndarray:
    out = Image.fromarray(image.copy())
    if box is not None:
        draw = ImageDraw.Draw(out)
        for offset in range(2):
            draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
    return np.array(out)


def make_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32) * 0.72
    out[gt] = out[gt] * 0.45 + np.array([0, 255, 0], dtype=np.float32) * 0.55
    out[pred] = out[pred] * 0.45 + np.array([255, 0, 0], dtype=np.float32) * 0.55
    out[np.logical_and(gt, pred)] = np.array([0, 220, 60], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    err = np.zeros((*gt.shape, 3), dtype=np.uint8)
    err[np.logical_and(gt, pred)] = [0, 200, 0]
    err[np.logical_and(~gt, pred)] = [255, 60, 60]
    err[np.logical_and(gt, ~pred)] = [60, 130, 255]
    return err


def load_panel_arrays(paths: tuple[Path, Path, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_path, mask_path, pred_path = paths
    pred = load_binary_mask(pred_path, 0)
    gt = load_binary_mask(mask_path, 127)
    if gt.shape != pred.shape:
        gt = resize_binary(gt, pred.shape)
    image = Image.open(image_path).convert("RGB").resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
    return np.array(image), gt, pred


def save_panel(record: dict[str, object], paths: tuple[Path, Path, Path], out_path: Path):
    image, gt, pred = load_panel_arrays(paths)
    box = bbox_from_masks(gt, pred)
    image_box = draw_box(image, box, (255, 255, 0))
    gt_rgb = np.repeat((gt.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    pred_rgb = np.repeat((pred.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    overlay = draw_box(make_overlay(image, gt, pred), box, (255, 255, 0))
    error = draw_box(make_error_map(gt, pred), box, (255, 255, 0))
    panels = [image_box, gt_rgb, pred_rgb, overlay, error]
    titles = [
        "Image",
        "GT",
        "Prediction",
        "Overlay",
        "Error: TP/FP/FN",
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), dpi=220)
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"{record['dataset']} | {record['name']} | IoU={record['iou']:.3f}, "
        f"P={record['precision']:.3f}, R={record['recall']:.3f}, FP={record['fp_pixels']}, FN={record['fn_pixels']}",
        fontsize=8,
    )
    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def select_categories(records: list[dict[str, object]], top_k: int) -> dict[str, list[dict[str, object]]]:
    non_empty = [r for r in records if int(r["gt_pixels"]) > 0 or int(r["pred_pixels"]) > 0]
    return {
        "worst_iou": sorted(non_empty, key=lambda r: (float(r["iou"]), -int(r["fp_pixels"]) - int(r["fn_pixels"])))[:top_k],
        "false_negative": sorted(
            [r for r in non_empty if int(r["fn_pixels"]) > 0],
            key=lambda r: (int(r["missed_targets"]), int(r["fn_pixels"]), 1.0 - float(r["recall"])),
            reverse=True,
        )[:top_k],
        "false_positive": sorted(
            [r for r in non_empty if int(r["fp_pixels"]) > 0],
            key=lambda r: (int(r["false_alarm_components"]), int(r["fp_pixels"]), 1.0 - float(r["precision"])),
            reverse=True,
        )[:top_k],
        "mixed": sorted(
            [r for r in non_empty if int(r["fp_pixels"]) > 0 and int(r["fn_pixels"]) > 0],
            key=lambda r: ((1.0 - float(r["iou"])) * (int(r["fp_pixels"]) + int(r["fn_pixels"]))),
            reverse=True,
        )[:top_k],
    }


def save_grid(selected: list[dict[str, object]], panel_lookup: dict[str, tuple[Path, Path, Path]], out_path: Path):
    if not selected:
        return
    cols = len(selected)
    fig, axes = plt.subplots(5, cols, figsize=(2.2 * cols, 10), dpi=220)
    if cols == 1:
        axes = np.expand_dims(axes, 1)
    row_titles = ["Image", "GT", "Prediction", "Overlay", "Error"]
    for col, record in enumerate(selected):
        image, gt, pred = load_panel_arrays(panel_lookup[str(record["name"])])
        box = bbox_from_masks(gt, pred)
        panels = [
            draw_box(image, box, (255, 255, 0)),
            np.repeat((gt.astype(np.uint8) * 255)[:, :, None], 3, axis=2),
            np.repeat((pred.astype(np.uint8) * 255)[:, :, None], 3, axis=2),
            draw_box(make_overlay(image, gt, pred), box, (255, 255, 0)),
            draw_box(make_error_map(gt, pred), box, (255, 255, 0)),
        ]
        for row, panel in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(row_titles[row], fontsize=8)
            if row == 0:
                axes[row, col].set_title(f"{record['name']}\nIoU={record['iou']:.3f}", fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_records(records: list[dict[str, object]], out_dir: Path):
    fields = [
        "dataset",
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
        "image_path",
        "mask_path",
    ]
    with open(out_dir / "ranking.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda r: float(r["iou"])):
            writer.writerow({field: record.get(field, "") for field in fields})


def write_summary(records: list[dict[str, object]], categories: dict[str, list[dict[str, object]]], out_dir: Path):
    ious = np.array([float(r["iou"]) for r in records], dtype=np.float64)
    fp = np.array([int(r["fp_pixels"]) for r in records], dtype=np.int64)
    fn = np.array([int(r["fn_pixels"]) for r in records], dtype=np.int64)
    summary = {
        "num_samples": len(records),
        "mean_iou": float(ious.mean()) if len(ious) else 0.0,
        "median_iou": float(np.median(ious)) if len(ious) else 0.0,
        "min_iou": float(ious.min()) if len(ious) else 0.0,
        "num_iou_below_0_5": int((ious < 0.5).sum()),
        "num_iou_below_0_7": int((ious < 0.7).sum()),
        "num_with_false_positive": int((fp > 0).sum()),
        "num_with_false_negative": int((fn > 0).sum()),
        "categories": {
            key: [
                {
                    "name": str(record["name"]),
                    "iou": float(record["iou"]),
                    "precision": float(record["precision"]),
                    "recall": float(record["recall"]),
                    "fp_pixels": int(record["fp_pixels"]),
                    "fn_pixels": int(record["fn_pixels"]),
                    "missed_targets": int(record["missed_targets"]),
                    "false_alarm_components": int(record["false_alarm_components"]),
                }
                for record in value
            ]
            for key, value in categories.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.config:
        records, panel_lookup = run_checkpoint_mode(args, out_dir)
    else:
        missing = [name for name in ["image_root", "mask_root", "split_file"] if getattr(args, name) is None]
        if missing:
            raise ValueError(f"Missing arguments for --pred-dir mode: {', '.join('--' + name.replace('_', '-') for name in missing)}")
        records, panel_lookup = run_saved_prediction_mode(args)

    write_records(records, out_dir)
    categories = select_categories(records, args.top_k)
    for category, selected in categories.items():
        for rank, record in enumerate(selected, start=1):
            name = str(record["name"])
            save_panel(record, panel_lookup[name], out_dir / category / f"{rank:02d}_{safe_name(name)}.png")
    grid_records: list[dict[str, object]] = []
    seen: set[str] = set()
    for category in ["worst_iou", "false_negative", "false_positive", "mixed"]:
        for record in categories[category]:
            name = str(record["name"])
            if name not in seen:
                grid_records.append(record)
                seen.add(name)
            if len(grid_records) >= args.grid_count:
                break
        if len(grid_records) >= args.grid_count:
            break
    save_grid(grid_records, panel_lookup, out_dir / "paper_grid.png")
    write_summary(records, categories, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "num_samples": len(records), "grid_count": len(grid_records)}, indent=2))


if __name__ == "__main__":
    main()
