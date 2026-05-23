from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from spdnet.models import SPD
from spdnet.utils.config import load_config, merge_overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Run SPD inference on images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--cfg-options", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), args.cfg_options)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    model = SPD(**cfg.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if state_dict and all(k.startswith("decode_head.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("decode_head."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    if args.deploy and hasattr(model.image_encoder, "switch_to_deploy"):
        model.eval()
        model.image_encoder.switch_to_deploy()
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((int(cfg.data.image_size), int(cfg.data.image_size)), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.normalize_mean, cfg.data.normalize_std),
    ])
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffixes = tuple(s.lower().lstrip(".") for s in cfg.data.suffixes)
    paths = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower().lstrip(".") in suffixes)
    with torch.no_grad():
        for path in tqdm(paths, desc="infer"):
            image = Image.open(path).convert("RGB")
            original_size = image.size
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            prob = torch.softmax(logits, dim=1)[0, 1]
            pred = (prob > float(cfg.eval.threshold)).cpu().numpy().astype(np.uint8) * 255
            pred_image = Image.fromarray(pred).resize(original_size, Image.NEAREST)
            pred_image.save(out_dir / path.name)


if __name__ == "__main__":
    main()
