from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.visualize_failure_cases import bbox_from_masks, load_binary_mask, make_overlay, resize_binary


def fixed_center_box(cx: int, cy: int, shape: tuple[int, int], size: int = 64):
    h, w = shape
    half = size // 2
    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(x0 + size, w)
    y1 = min(y0 + size, h)
    x0 = max(x1 - size, 0)
    y0 = max(y1 - size, 0)
    return int(x0), int(y0), int(x1), int(y1)


def largest_component_center(mask: np.ndarray) -> tuple[int, int] | None:
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    regions = measure.regionprops(labels)
    if not regions:
        return None
    region = max(regions, key=lambda item: item.area)
    cy, cx = region.centroid
    return int(round(cx)), int(round(cy))


def choose_roi(gt: np.ndarray, pred: np.ndarray, mode: str) -> tuple[int, int, int, int]:
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(gt, ~pred)
    if mode == "fp":
        center = largest_component_center(fp) or largest_component_center(pred) or largest_component_center(gt)
    elif mode == "fn":
        center = largest_component_center(fn) or largest_component_center(gt) or largest_component_center(pred)
    else:
        center = largest_component_center(np.logical_or(fp, fn)) or largest_component_center(gt) or largest_component_center(pred)
    if center is None:
        center = (gt.shape[1] // 2, gt.shape[0] // 2)
    return fixed_center_box(center[0], center[1], gt.shape, size=64)


def draw_roi(image: np.ndarray, box: tuple[int, int, int, int], color=(255, 255, 0), width: int = 3):
    out = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(out)
    for offset in range(width):
        draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
    return np.array(out)


def load_case(image_path: str, mask_path: str, pred_path: str, mode: str):
    pred = load_binary_mask(Path(pred_path), 0)
    gt = load_binary_mask(Path(mask_path), 127)
    if gt.shape != pred.shape:
        gt = resize_binary(gt, pred.shape)
    image = Image.open(image_path).convert("RGB").resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
    image_arr = np.array(image)
    overlay = make_overlay(image_arr, gt, pred)
    crop_box = choose_roi(gt, pred, mode)
    x0, y0, x1, y1 = crop_box
    return draw_roi(image_arr, crop_box), overlay[y0:y1, x0:x1]


def save_case(filename: str, image_path: str, mask_path: str, pred_path: str, mode: str, out_dir: Path, font_prop):
    full_image, overlay_crop = load_case(image_path, mask_path, pred_path, mode)
    out_png = out_dir / f"{filename}.png"
    out_pdf = out_dir / f"{filename}.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=600)
    axes[0].imshow(full_image)
    axes[1].imshow(overlay_crop, interpolation="nearest")
    for ax, label in zip(axes, ["Image", "Zoomed Overlay"]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(label, fontproperties=font_prop, fontsize=11, labelpad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.08, wspace=0.03)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    Image.open(out_png).save(out_png, dpi=(600, 600))


def main():
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    font_manager.fontManager.addfont(font_path)
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 11

    out_dir = Path("refs/failure_case_figures/publication_ready")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            "irtiny_001232_pub",
            "data/IRTiny-BD-10K/val/images/001232.png",
            "data/IRTiny-BD-10K/val/masks/001232.png",
            "work_dirs/failure_cases/irtiny/predictions/001232.png",
            "mixed",
        ),
        (
            "irtiny_000561_pub",
            "data/IRTiny-BD-10K/val/images/000561.png",
            "data/IRTiny-BD-10K/val/masks/000561.png",
            "work_dirs/failure_cases/irtiny/predictions/000561.png",
            "fp",
        ),
        (
            "ch3_sky_airplane107_pub",
            "data/ch3/mulframe/images/sky-1_20240905094212_Circular_airplane1/107.png",
            "data/ch3/mulframe/masks/sky-1_20240905094212_Circular_airplane1/107.png",
            "work_dirs/failure_cases/ch3/predictions/sky-1_20240905094212_Circular_airplane1_107.png",
            "fn",
        ),
        (
            "ch3_mountain154_pub",
            "data/ch3/mulframe/images/mountian-27_20240918153100_Bezier Curve_Point1/154.png",
            "data/ch3/mulframe/masks/mountian-27_20240918153100_Bezier Curve_Point1/154.png",
            "work_dirs/failure_cases/ch3/predictions/mountian-27_20240918153100_Bezier Curve_Point1_154.png",
            "fp",
        ),
    ]

    for case in cases:
        save_case(*case, out_dir=out_dir, font_prop=font_prop)

    print(out_dir.resolve())


if __name__ == "__main__":
    main()
