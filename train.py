from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from spdnet.data import build_dataloader
from spdnet.losses import SegmentationLoss, token_contrastive_loss
from spdnet.metrics import ABCForegroundIoUMetrics, ABCMeanIoUMetrics, BinarySegmentationMetrics, PaperMetrics
from spdnet.models import SPD
from spdnet.optim import build_optimizer, build_scheduler
from spdnet.utils.config import load_config, merge_overrides
from spdnet.utils.reproducibility import seed_everything

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is optional at runtime
    SummaryWriter = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPD on infrared binary segmentation datasets.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume")
    parser.add_argument("--load-from", default=None, help="Checkpoint weights to initialize")
    parser.add_argument("--work-dir", default=None, help="Override output directory")
    parser.add_argument("--cfg-options", nargs="*", default=None, help="Override config keys, e.g. data.root=/path train.epochs=100")
    return parser.parse_args()


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    distributed = local_rank >= 0 and torch.cuda.is_available()
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, local_rank, rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def checkpoint_state(path: str | Path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


def normalize_state_dict(state_dict):
    if state_dict and all(k.startswith("decode_head.") for k in state_dict.keys()):
        return {k.removeprefix("decode_head."): v for k, v in state_dict.items()}
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


def reduce_mean(value: float, device: torch.device, distributed: bool) -> float:
    tensor = torch.tensor(float(value), device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def log_line(path: Path, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def draw_curves(history: list[dict], work_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r.get("val_loss") for r in history]
    miou = [r.get("val_mIoU") for r in history]
    niou = [r.get("val_threshold_nIoU") for r in history]
    f1 = [r.get("val_threshold_F1") for r in history]

    def plot(name: str, series: list[tuple[str, list[float | None]]], ylabel: str):
        plt.figure()
        for label, values in series:
            xs = [e for e, v in zip(epochs, values) if v is not None]
            ys = [v for v in values if v is not None]
            if ys:
                plt.plot(xs, ys, label=label)
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel("Epoch")
        plt.savefig(work_dir / name)
        plt.close()

    plot("fig_loss.png", [("train_loss", train_loss), ("test_loss", val_loss)], "Loss")
    plot("fig_IoU.png", [("mIoU", miou), ("nIoU", niou)], "IoU")
    plot("fig_F1-score.png", [("F1-score", f1)], "F1-score")


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold, rank: int = 0):
    model.eval()
    abc_metrics = ABCForegroundIoUMetrics()
    argmax_metrics = ABCMeanIoUMetrics(num_classes=2)
    threshold_metrics = BinarySegmentationMetrics(threshold=threshold)
    paper_metrics = PaperMetrics()
    losses = []
    iterator = tqdm(loader, desc="val", leave=False) if is_main_process(rank) else loader
    for images, masks, _ in iterator:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        losses.append(loss.item())
        abc_metrics.update(logits, masks)
        argmax_metrics.update(logits, masks)
        threshold_metrics.update(logits, masks)
        paper_metrics.update(logits, masks)
    result = abc_metrics.compute()
    result.update(argmax_metrics.compute())
    threshold_result = threshold_metrics.compute()
    result.update({f"threshold_{k}": v for k, v in threshold_result.items()})
    paper_result = paper_metrics.compute()
    result.update({f"paper_{k}": v for k, v in paper_result.items()})
    result["loss"] = float(sum(losses) / max(1, len(losses)))
    return result


def main():
    args = parse_args()
    distributed, local_rank, rank, world_size, device = init_distributed()
    cfg = merge_overrides(load_config(args.config), args.cfg_options)
    seed_everything(int(cfg.seed) + rank, bool(cfg.deterministic))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = Path(args.work_dir or Path(cfg.train.save_dir) / cfg.experiment_name)
    if is_main_process(rank):
        work_dir.mkdir(parents=True, exist_ok=True)
        with open(work_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        log_line(work_dir / "train_log.txt", f"time: {timestamp}")
        log_line(work_dir / "train_log.txt", json.dumps(cfg, indent=2, ensure_ascii=False))
    if distributed:
        dist.barrier()

    train_loader = build_dataloader(cfg, "train", training=True, distributed=distributed)
    val_loader = build_dataloader(cfg, "val", training=False, distributed=False)
    model = SPD(**cfg.model).to(device)
    if args.load_from:
        model.load_state_dict(normalize_state_dict(checkpoint_state(args.load_from, device)), strict=True)

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    criterion = SegmentationLoss(**{k: v for k, v in cfg.train.loss.items() if k in ("ce_weight", "dice_weight")})
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp) and device.type == "cuda")
    writer = SummaryWriter(log_dir=str(work_dir / "tf_logs")) if is_main_process(rank) and SummaryWriter is not None else None

    start_epoch = 1
    best_iou = -1.0
    history = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(normalize_state_dict(ckpt["state_dict"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", best_iou)
        history = ckpt.get("history", history)

    for epoch in range(start_epoch, int(cfg.train.epochs) + 1):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        lr = scheduler.step(epoch)
        train_losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.train.epochs}") if is_main_process(rank) else train_loader
        for step, (images, masks, _) in enumerate(pbar, start=1):
            since = time.time()
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp) and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
                loss = loss + float(cfg.train.loss.token_contrast_weight) * token_contrastive_loss(model)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            mean_loss = sum(train_losses) / len(train_losses)
            if is_main_process(rank):
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{lr:.6g}")
                interval = int(getattr(cfg.train, "log_interval", 10))
                if step % interval == 0:
                    log_line(
                        work_dir / "train_log.txt",
                        "{} Epoch: [{}/{}] Iter[{}/{}] Loss: {:.4f} Lr: {:.5f} Time: {:.5f}".format(
                            datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                            epoch,
                            cfg.train.epochs,
                            step,
                            len(train_loader),
                            mean_loss,
                            lr,
                            time.time() - since,
                        ),
                    )

        train_loss = reduce_mean(sum(train_losses) / max(1, len(train_losses)), device, distributed)
        record = {"epoch": epoch, "lr": lr, "train_loss": train_loss}
        is_best = False
        if epoch % int(cfg.train.val_interval) == 0:
            val_result = evaluate(unwrap_model(model), val_loader, criterion, device, float(cfg.eval.threshold), rank=rank) if is_main_process(rank) else None
            if distributed:
                dist.barrier()
            if is_main_process(rank):
                record.update({f"val_{k}": v for k, v in val_result.items()})
                is_best = val_result["mIoU"] > best_iou
                best_iou = max(best_iou, val_result["mIoU"])
                log_line(
                    work_dir / "train_log.txt",
                    "{} Epoch: [{}/{}] Loss: {:.4f} mIoU: {:.4f} nIoU: {:.4f} F1-score: {:.4f} Best_mIoU: {:.4f}".format(
                        datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                        epoch,
                        cfg.train.epochs,
                        val_result["loss"],
                        val_result["mIoU"],
                        val_result["threshold_nIoU"],
                        val_result["threshold_F1"],
                        best_iou,
                    ),
                )
        elif distributed:
            dist.barrier()

        if is_main_process(rank):
            history.append(record)
            print(json.dumps(record, ensure_ascii=False, indent=2))
            state_dict = unwrap_model(model).state_dict()
            ckpt = {
                "epoch": epoch,
                "state_dict": state_dict,
                "optimizer": optimizer.state_dict(),
                "best_iou": best_iou,
                "history": history,
                "config": cfg,
            }
            torch.save(ckpt, work_dir / "last.pth")
            torch.save(ckpt, work_dir / "last.pth.tar")
            if is_best:
                torch.save(ckpt, work_dir / "best.pth")
                torch.save(ckpt, work_dir / "best.pth.tar")
            with open(work_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            draw_curves(history, work_dir)
            if writer is not None:
                writer.add_scalar("train/train_loss", train_loss, epoch)
                writer.add_scalar("train/lr", lr, epoch)
                if "val_loss" in record:
                    writer.add_scalar("train/test_loss", record["val_loss"], epoch)
                    writer.add_scalar("test/mIoU", record["val_mIoU"], epoch)
                    writer.add_scalar("test/nIoU", record["val_threshold_nIoU"], epoch)
                    writer.add_scalar("test/F1-score", record["val_threshold_F1"], epoch)
                writer.flush()
        if distributed:
            dist.barrier()

    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
