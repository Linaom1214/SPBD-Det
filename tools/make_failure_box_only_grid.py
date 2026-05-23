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


def regions(mask: np.ndarray):
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    return sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)


def region_box(region, pad: int, shape: tuple[int, int]):
    minr, minc, maxr, maxc = region.bbox
    h, w = shape
    return max(minc - pad, 0), max(minr - pad, 0), min(maxc + pad, w), min(maxr + pad, h)


def draw_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int]], color: tuple[int, int, int], width: int):
    out = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(out)
    for box in boxes:
        for offset in range(width):
            draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
    return np.array(out)


def draw_case(image_path: str, mask_path: str, pred_path: str, max_fp_boxes: int = 8):
    pred = load_binary_mask(Path(pred_path), 0)
    gt = load_binary_mask(Path(mask_path), 127)
    if gt.shape != pred.shape:
        gt = resize_binary(gt, pred.shape)
    image = Image.open(image_path).convert("RGB").resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
    image_arr = np.array(image)

    fp = np.logical_and(pred, ~gt)
    gt_boxes = [region_box(r, pad=8, shape=gt.shape) for r in regions(gt)]
    fp_boxes = [region_box(r, pad=8, shape=gt.shape) for r in regions(fp)[:max_fp_boxes]]

    vis = draw_boxes(image_arr, fp_boxes, FP_COLOR, width=2)
    vis = draw_boxes(vis, gt_boxes, GT_COLOR, width=3)
    return vis


def main():
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    font_manager.fontManager.addfont(font_path)
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 11

    out_dir = Path("refs/failure_case_figures/box_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            "(a) IRTiny-BD-10K",
            "data/IRTiny-BD-10K/val/images/001232.png",
            "data/IRTiny-BD-10K/val/masks/001232.png",
            "work_dirs/failure_cases/irtiny/predictions/001232.png",
        ),
        (
            "(b) IRTiny-BD-10K",
            "data/IRTiny-BD-10K/val/images/000561.png",
            "data/IRTiny-BD-10K/val/masks/000561.png",
            "work_dirs/failure_cases/irtiny/predictions/000561.png",
        ),
        (
            "(c) IRReversal/ch3",
            "data/ch3/mulframe/images/sky-1_20240905094212_Circular_airplane1/107.png",
            "data/ch3/mulframe/masks/sky-1_20240905094212_Circular_airplane1/107.png",
            "work_dirs/failure_cases/ch3/predictions/sky-1_20240905094212_Circular_airplane1_107.png",
        ),
        (
            "(d) IRReversal/ch3",
            "data/ch3/mulframe/images/mountian-27_20240918153100_Bezier Curve_Point1/154.png",
            "data/ch3/mulframe/masks/mountian-27_20240918153100_Bezier Curve_Point1/154.png",
            "work_dirs/failure_cases/ch3/predictions/mountian-27_20240918153100_Bezier Curve_Point1_154.png",
        ),
    ]

    panels = [draw_case(image_path, mask_path, pred_path) for _, image_path, mask_path, pred_path in cases]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), dpi=600)
    for ax, panel, (label, *_rest) in zip(axes.ravel(), panels, cases):
        ax.imshow(panel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(label, fontproperties=font_prop, fontsize=11, labelpad=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")

    fig.text(0.34, 0.015, "Green box: target", fontproperties=font_prop, fontsize=11, color="green", ha="center")
    fig.text(0.66, 0.015, "Red box: false alarm", fontproperties=font_prop, fontsize=11, color="red", ha="center")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.06, wspace=0.025, hspace=0.09)

    png = out_dir / "failure_box_only_grid.png"
    pdf = out_dir / "failure_box_only_grid.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    Image.open(png).save(png, dpi=(600, 600))
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
