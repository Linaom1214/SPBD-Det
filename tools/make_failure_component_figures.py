from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.visualize_failure_cases import load_binary_mask, resize_binary


GT_COLOR = (0, 255, 80)
FP_COLOR = (255, 40, 40)
PRED_COLOR = (255, 210, 0)


def fixed_box(cx: int, cy: int, shape: tuple[int, int], size: int = 72):
    h, w = shape
    half = size // 2
    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(x0 + size, w)
    y1 = min(y0 + size, h)
    x0 = max(x1 - size, 0)
    y0 = max(y1 - size, 0)
    return int(x0), int(y0), int(x1), int(y1)


def regions(mask: np.ndarray):
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    return sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)


def region_box(region, pad: int, shape: tuple[int, int]):
    minr, minc, maxr, maxc = region.bbox
    h, w = shape
    return max(minc - pad, 0), max(minr - pad, 0), min(maxc + pad, w), min(maxr + pad, h)


def region_center(region):
    cy, cx = region.centroid
    return int(round(cx)), int(round(cy))


def draw_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int]], color: tuple[int, int, int], width: int):
    out = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(out)
    for box in boxes:
        for offset in range(width):
            draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
    return np.array(out)


def draw_mask_contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], width: int = 2):
    out = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(out)
    for region in regions(mask):
        box = region_box(region, pad=1, shape=mask.shape)
        for offset in range(width):
            draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
    return np.array(out)


def crop_with_masks(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1].copy()
    crop_gt = gt[y0:y1, x0:x1]
    crop_pred = pred[y0:y1, x0:x1]
    vis = crop.astype(np.float32) * 0.78
    vis[crop_gt] = vis[crop_gt] * 0.25 + np.array(GT_COLOR, dtype=np.float32) * 0.75
    vis[np.logical_and(crop_pred, ~crop_gt)] = vis[np.logical_and(crop_pred, ~crop_gt)] * 0.25 + np.array(FP_COLOR, dtype=np.float32) * 0.75
    vis[np.logical_and(crop_pred, crop_gt)] = vis[np.logical_and(crop_pred, crop_gt)] * 0.25 + np.array(PRED_COLOR, dtype=np.float32) * 0.75
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    vis = draw_mask_contour(vis, crop_gt, GT_COLOR, width=2)
    vis = draw_mask_contour(vis, crop_pred, FP_COLOR, width=1)
    return vis


def load_case(image_path: str, mask_path: str, pred_path: str):
    pred = load_binary_mask(Path(pred_path), 0)
    gt = load_binary_mask(Path(mask_path), 127)
    if gt.shape != pred.shape:
        gt = resize_binary(gt, pred.shape)
    image = Image.open(image_path).convert("RGB").resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
    return np.array(image), gt, pred


def make_case_panels(image_path: str, mask_path: str, pred_path: str):
    image, gt, pred = load_case(image_path, mask_path, pred_path)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(gt, ~pred)
    gt_regions = regions(gt)
    fp_regions = regions(fp)
    pred_regions = regions(pred)

    gt_boxes = [region_box(r, pad=8, shape=gt.shape) for r in gt_regions[:4]]
    fp_boxes = [region_box(r, pad=8, shape=gt.shape) for r in fp_regions[:8]]
    full = draw_boxes(image, gt_boxes, GT_COLOR, width=3)
    full = draw_boxes(full, fp_boxes, FP_COLOR, width=2)

    if gt_regions:
        cx, cy = region_center(gt_regions[0])
    elif pred_regions:
        cx, cy = region_center(pred_regions[0])
    else:
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
    target_box = fixed_box(cx, cy, gt.shape, size=72)

    if fp_regions:
        cx, cy = region_center(fp_regions[0])
    elif pred_regions:
        cx, cy = region_center(pred_regions[0])
    elif gt_regions:
        cx, cy = region_center(gt_regions[0])
    else:
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
    fp_box = fixed_box(cx, cy, gt.shape, size=72)

    full = draw_boxes(full, [target_box], GT_COLOR, width=4)
    if fp_regions:
        full = draw_boxes(full, [fp_box], FP_COLOR, width=4)

    target_zoom = crop_with_masks(image, gt, pred, target_box)
    fp_zoom = crop_with_masks(image, gt, pred, fp_box)
    return full, target_zoom, fp_zoom


def save_individual_case(name: str, image_path: str, mask_path: str, pred_path: str, out_dir: Path, font_prop):
    full, target_zoom, fp_zoom = make_case_panels(image_path, mask_path, pred_path)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.35), dpi=600)
    for ax, panel, label in zip(axes, [full, target_zoom, fp_zoom], ["Image", "Target zoom", "False-alarm zoom"]):
        ax.imshow(panel, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(label, fontproperties=font_prop, fontsize=10, labelpad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_edgecolor("black")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.11, wspace=0.025)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(pdf, dpi=600, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    Image.open(png).save(png, dpi=(600, 600))
    return full, target_zoom, fp_zoom


def save_grid(cases, out_dir: Path, font_prop):
    panels = []
    for _, image_path, mask_path, pred_path in cases:
        panels.append(make_case_panels(image_path, mask_path, pred_path))
    fig, axes = plt.subplots(len(cases), 3, figsize=(7.5, 8.8), dpi=600)
    col_labels = ["Image", "Target zoom", "False-alarm zoom"]
    row_labels = ["(a)", "(b)", "(c)", "(d)"]
    for row, row_panels in enumerate(panels):
        for col, panel in enumerate(row_panels):
            ax = axes[row, col]
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == len(cases) - 1:
                ax.set_xlabel(col_labels[col], fontproperties=font_prop, fontsize=10, labelpad=2)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontproperties=font_prop, fontsize=11, rotation=0, labelpad=12, va="center")
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_edgecolor("black")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.995, bottom=0.035, wspace=0.025, hspace=0.035)
    png = out_dir / "failure_component_grid.png"
    pdf = out_dir / "failure_component_grid.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(pdf, dpi=600, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    Image.open(png).save(png, dpi=(600, 600))


def main():
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    font_manager.fontManager.addfont(font_path)
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 10

    out_dir = Path("refs/failure_case_figures/component_view")
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        (
            "irtiny_001232_component",
            "data/IRTiny-BD-10K/val/images/001232.png",
            "data/IRTiny-BD-10K/val/masks/001232.png",
            "work_dirs/failure_cases/irtiny/predictions/001232.png",
        ),
        (
            "irtiny_000561_component",
            "data/IRTiny-BD-10K/val/images/000561.png",
            "data/IRTiny-BD-10K/val/masks/000561.png",
            "work_dirs/failure_cases/irtiny/predictions/000561.png",
        ),
        (
            "ch3_sky_airplane107_component",
            "data/ch3/mulframe/images/sky-1_20240905094212_Circular_airplane1/107.png",
            "data/ch3/mulframe/masks/sky-1_20240905094212_Circular_airplane1/107.png",
            "work_dirs/failure_cases/ch3/predictions/sky-1_20240905094212_Circular_airplane1_107.png",
        ),
        (
            "ch3_mountain154_component",
            "data/ch3/mulframe/images/mountian-27_20240918153100_Bezier Curve_Point1/154.png",
            "data/ch3/mulframe/masks/mountian-27_20240918153100_Bezier Curve_Point1/154.png",
            "work_dirs/failure_cases/ch3/predictions/mountian-27_20240918153100_Bezier Curve_Point1_154.png",
        ),
    ]
    for case in cases:
        save_individual_case(*case, out_dir=out_dir, font_prop=font_prop)
    save_grid(cases, out_dir, font_prop)
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
