from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets = targets.squeeze(1).float()
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        return (1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)).mean()


class SegmentationLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = BinaryDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_labels = targets.squeeze(1).long()
        return self.ce_weight * self.ce(logits, target_labels) + self.dice_weight * self.dice(logits, targets)


def token_contrastive_loss(model: nn.Module) -> torch.Tensor:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    decoder = model.mask_decoder
    if not hasattr(decoder, "indices"):
        return next(model.parameters()).new_tensor(0.0)
    loss_fn = nn.CosineEmbeddingLoss(margin=0.0)
    loss = next(model.parameters()).new_tensor(0.0)
    for block_index in decoder.indices:
        tokens = getattr(decoder, f"token_s{block_index}").weight
        bright = tokens[0].unsqueeze(0)
        dark = tokens[1].unsqueeze(0)
        target = -torch.ones(bright.size(0), device=bright.device)
        loss = loss + loss_fn(bright, dark, target)
    return loss
