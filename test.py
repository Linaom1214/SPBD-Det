from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from spdnet.data import build_dataloader
from spdnet.losses import SegmentationLoss
from spdnet.metrics import ABCForegroundIoUMetrics, ABCMeanIoUMetrics, BinarySegmentationMetrics, PaperMetrics
from spdnet.models import SPD
from spdnet.utils.config import load_config, merge_overrides
from spdnet.utils.reproducibility import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SPD.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--save-pred", action="store_true", help="Save binary predictions")
    parser.add_argument("--deploy", action="store_true", help="Fuse re-parameterized encoder branches before evaluation")
    parser.add_argument("--abc-legacy", action="store_true", help="Report ABC checkpoint-stored legacy metrics as primary results")
    parser.add_argument("--cfg-options", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), args.cfg_options)
    seed_everything(int(cfg.seed), bool(cfg.deterministic))
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    loader = build_dataloader(cfg, args.split, training=False)
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
    criterion = SegmentationLoss(**{k: v for k, v in cfg.train.loss.items() if k in ("ce_weight", "dice_weight")})
    abc_metrics = ABCForegroundIoUMetrics()
    argmax_metrics = ABCMeanIoUMetrics(num_classes=2)
    threshold_metrics = BinarySegmentationMetrics(threshold=float(cfg.eval.threshold))
    paper_metrics = PaperMetrics()
    losses = []
    work_dir = Path(args.work_dir or Path(args.checkpoint).parent / f"eval_{args.split}")
    pred_dir = work_dir / "predictions"
    if args.save_pred:
        pred_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for images, masks, names in tqdm(loader, desc=f"eval-{args.split}"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            losses.append(criterion(logits, masks).item())
            abc_metrics.update(logits, masks)
            argmax_metrics.update(logits, masks)
            threshold_metrics.update(logits, masks)
            paper_metrics.update(logits, masks)
            if args.save_pred:
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = (probs > float(cfg.eval.threshold)).cpu().numpy().astype(np.uint8) * 255
                for pred, name in zip(preds, names):
                    Image.fromarray(pred).save(pred_dir / name)
    result = abc_metrics.compute()
    result.update(argmax_metrics.compute())
    threshold_result = threshold_metrics.compute()
    result.update({f"threshold_{k}": v for k, v in threshold_result.items()})
    paper_result = paper_metrics.compute()
    result.update({f"paper_{k}": v for k, v in paper_result.items()})
    result["loss"] = float(sum(losses) / max(1, len(losses)))
    if args.abc_legacy and isinstance(ckpt, dict):
        single_pass_result = result.copy()
        legacy_result = {}
        if "mIoU" in ckpt:
            legacy_result["mIoU"] = float(ckpt["mIoU"])
        if "nIoU" in ckpt:
            legacy_result["nIoU"] = float(ckpt["nIoU"])
        if "f1" in ckpt:
            legacy_result["F1"] = float(ckpt["f1"])
        if "loss" in ckpt:
            legacy_result["loss"] = float(ckpt["loss"])
        legacy_result.update({f"single_pass_{k}": v for k, v in single_pass_result.items()})
        result = legacy_result
    result["threshold"] = float(cfg.eval.threshold)
    result["postprocess"] = cfg.eval.postprocess
    with open(work_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
