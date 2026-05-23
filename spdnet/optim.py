from __future__ import annotations

import torch


def build_optimizer(model: torch.nn.Module, cfg):
    train_cfg = cfg.train
    name = train_cfg.optimizer.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=tuple(train_cfg.betas))
    if name == "adam":
        return torch.optim.Adam(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=tuple(train_cfg.betas))
    if name == "sgd":
        return torch.optim.SGD(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {train_cfg.optimizer}")


class PolyWarmupLR:
    def __init__(self, optimizer, max_epochs: int, base_lr: float, power: float = 0.9, min_lr: float = 1e-4, warmup_epochs: int = 5):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.power = power
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs

    def step(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = max(self.min_lr, self.base_lr * (1.0 - progress) ** self.power)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def build_scheduler(optimizer, cfg):
    sched_cfg = cfg.train.scheduler
    if sched_cfg.type.lower() == "poly":
        return PolyWarmupLR(
            optimizer,
            max_epochs=cfg.train.epochs,
            base_lr=cfg.train.lr,
            power=sched_cfg.power,
            min_lr=sched_cfg.min_lr,
            warmup_epochs=sched_cfg.warmup_epochs,
        )
    raise ValueError(f"Unsupported scheduler: {sched_cfg.type}")
