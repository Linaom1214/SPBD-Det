from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.visualize_failure_cases import load_binary_mask, make_overlay, resize_binary


def main():
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    font_manager.fontManager.addfont(font_path)
    times_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = times_prop.get_name()
    plt.rcParams["font.size"] = 8

    out_dir = Path("refs/failure_case_figures/image_overlay_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            "irtiny_001232_image_overlay_only.png",
            "data/IRTiny-BD-10K/val/images/001232.png",
            "data/IRTiny-BD-10K/val/masks/001232.png",
            "work_dirs/failure_cases/irtiny/predictions/001232.png",
        ),
        (
            "irtiny_000561_image_overlay_only.png",
            "data/IRTiny-BD-10K/val/images/000561.png",
            "data/IRTiny-BD-10K/val/masks/000561.png",
            "work_dirs/failure_cases/irtiny/predictions/000561.png",
        ),
        (
            "ch3_sky_airplane107_image_overlay_only.png",
            "data/ch3/mulframe/images/sky-1_20240905094212_Circular_airplane1/107.png",
            "data/ch3/mulframe/masks/sky-1_20240905094212_Circular_airplane1/107.png",
            "work_dirs/failure_cases/ch3/predictions/sky-1_20240905094212_Circular_airplane1_107.png",
        ),
        (
            "ch3_mountain154_image_overlay_only.png",
            "data/ch3/mulframe/images/mountian-27_20240918153100_Bezier Curve_Point1/154.png",
            "data/ch3/mulframe/masks/mountian-27_20240918153100_Bezier Curve_Point1/154.png",
            "work_dirs/failure_cases/ch3/predictions/mountian-27_20240918153100_Bezier Curve_Point1_154.png",
        ),
    ]

    for filename, image_path, mask_path, pred_path in cases:
        pred = load_binary_mask(Path(pred_path), 0)
        gt = load_binary_mask(Path(mask_path), 127)
        if gt.shape != pred.shape:
            gt = resize_binary(gt, pred.shape)
        image = Image.open(image_path).convert("RGB").resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
        image_arr = np.array(image)
        overlay = make_overlay(image_arr, gt, pred)

        fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.7), dpi=300)
        for ax, panel, label in zip(axes, [image_arr, overlay], ["Image", "Overlay"]):
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(label, fontproperties=times_prop, fontsize=8, labelpad=1)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0.12, wspace=0.015)
        fig.savefig(out_dir / filename, bbox_inches="tight", pad_inches=0.005)
        plt.close(fig)

    print(out_dir.resolve())


if __name__ == "__main__":
    main()
