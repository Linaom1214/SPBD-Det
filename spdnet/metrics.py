from __future__ import annotations

import numpy as np
import torch
from skimage import measure


class ABCForegroundIoUMetrics:
    """Reproduce ABC training metric for 2-channel SPD outputs: argmax then foreground IoU."""

    def reset(self) -> None:
        self.total_inter = 0.0
        self.total_union = 0.0
        self.total_correct = 0.0
        self.total_label = 0.0

    def __init__(self):
        self.reset()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, masks: torch.Tensor) -> None:
        if logits.shape[1] == 1:
            predict = (torch.sigmoid(logits[:, 0]) > 0.5).float()
        else:
            predict = (logits.argmax(dim=1) == 1).float()
        target = masks.squeeze(1).float()
        correct = (((predict == target).float()) * (target > 0).float()).sum().item()
        labeled = (target > 0).float().sum().item()
        intersection = predict * ((predict == target).float())
        area_inter, _ = np.histogram(intersection.detach().cpu().numpy(), bins=1, range=(1, 1))
        area_pred, _ = np.histogram(predict.detach().cpu().numpy(), bins=1, range=(1, 1))
        area_lab, _ = np.histogram(target.detach().cpu().numpy(), bins=1, range=(1, 1))
        area_union = area_pred + area_lab - area_inter
        self.total_inter += float(area_inter[0])
        self.total_union += float(area_union[0])
        self.total_correct += float(correct)
        self.total_label += float(labeled)

    def compute(self) -> dict[str, float]:
        miou = self.total_inter / (np.spacing(1) + self.total_union)
        pix_acc = self.total_correct / (np.spacing(1) + self.total_label)
        return {
            "mIoU": float(miou),
            "PixelAcc": float(pix_acc),
        }


class ABCMeanIoUMetrics:
    def __init__(self, num_classes: int = 2, beta: float = 1.0):
        self.num_classes = num_classes
        self.beta = beta
        self.reset()

    def reset(self) -> None:
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, masks: torch.Tensor) -> None:
        if logits.shape[1] == 1:
            preds = (torch.sigmoid(logits[:, 0]) > 0.5).long()
        else:
            preds = logits.argmax(dim=1).long()
        targets = masks.squeeze(1).long()
        preds_np = preds.detach().cpu().numpy().reshape(-1)
        targets_np = targets.detach().cpu().numpy().reshape(-1)
        valid = (targets_np >= 0) & (targets_np < self.num_classes)
        hist = np.bincount(
            self.num_classes * targets_np[valid] + preds_np[valid],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += hist

    def compute(self) -> dict[str, float]:
        tp = np.diag(self.confusion)
        gts = self.confusion.sum(axis=1)
        preds = self.confusion.sum(axis=0)
        union = gts + preds - tp
        iou = tp / union
        precision = tp / preds
        recall = tp / gts
        f_score = (1 + self.beta ** 2) * (precision * recall) / ((self.beta ** 2 * precision) + recall)
        return {
            "argmax_mIoU": float(np.nanmean(iou)),
            "argmax_mPrecision": float(np.nanmean(precision)),
            "argmax_mRecall": float(np.nanmean(recall)),
            "argmax_mFscore": float(np.nanmean(f_score)),
            "argmax_background_IoU": float(iou[0]),
            "argmax_target_IoU": float(iou[1]) if self.num_classes > 1 else float("nan"),
        }


class BinarySegmentationMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.sample_ious: list[float] = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, masks: torch.Tensor) -> None:
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits[:, 0])
        else:
            probs = torch.softmax(logits, dim=1)[:, 1]
        preds = probs > self.threshold
        targets = masks.squeeze(1).bool()
        self.tp += torch.logical_and(preds, targets).sum().item()
        self.fp += torch.logical_and(preds, ~targets).sum().item()
        self.fn += torch.logical_and(~preds, targets).sum().item()
        self.tn += torch.logical_and(~preds, ~targets).sum().item()
        for pred, target in zip(preds, targets):
            inter = torch.logical_and(pred, target).sum().item()
            union = torch.logical_or(pred, target).sum().item()
            self.sample_ious.append(1.0 if union == 0 else inter / union)

    def compute(self) -> dict[str, float]:
        eps = np.spacing(1)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        pixel_acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        return {
            "IoU": float(iou),
            "nIoU": float(np.mean(self.sample_ious)) if self.sample_ious else 0.0,
            "F1": float(f1),
            "Precision": float(precision),
            "Recall": float(recall),
            "PixelAcc": float(pixel_acc),
        }


class PaperMetrics:
    """Match ABC test_pd.py paper metrics for pixAcc, mIoU, PD, and FA."""

    def __init__(self, match_distance: float = 3.0):
        self.match_distance = match_distance
        self.reset()

    def reset(self) -> None:
        self.total_inter = 0.0
        self.total_union = 0.0
        self.total_correct = 0.0
        self.total_label = 0.0
        self.dismatch_pixel = 0.0
        self.all_pixel = 0.0
        self.pd = 0.0
        self.targets = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, masks: torch.Tensor) -> None:
        if logits.shape[1] == 1:
            preds = torch.sigmoid(logits[:, 0]) > 0.5
        else:
            preds = logits.argmax(dim=1) > 0
        targets = masks.squeeze(1).float()
        preds_float = preds.float()
        self.total_correct += float((((preds_float == targets).float()) * (targets > 0).float()).sum().item())
        self.total_label += float((targets > 0).float().sum().item())
        intersection = preds_float * ((preds_float == targets).float())
        area_inter, _ = np.histogram(intersection.detach().cpu().numpy(), bins=1, range=(1, 1))
        area_pred, _ = np.histogram(preds_float.detach().cpu().numpy(), bins=1, range=(1, 1))
        area_lab, _ = np.histogram(targets.detach().cpu().numpy(), bins=1, range=(1, 1))
        area_union = area_pred + area_lab - area_inter
        self.total_inter += float(area_inter[0])
        self.total_union += float(area_union[0])
        preds_np = preds.detach().cpu().numpy().astype(np.uint8)
        targets_np = targets.detach().cpu().numpy().astype(np.int64)
        for pred, target in zip(preds_np, targets_np):
            self._update_pd_fa(pred, target)

    def _update_pd_fa(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred_regions = list(measure.regionprops(measure.label(pred, connectivity=2)))
        target_regions = list(measure.regionprops(measure.label(target, connectivity=2)))
        self.targets += len(target_regions)
        image_area_total = [np.array(region.area) for region in pred_regions]
        image_area_match = []
        distance_match = []
        for target_region in target_regions:
            target_centroid = np.array(list(target_region.centroid))
            for pred_index, pred_region in enumerate(pred_regions):
                pred_centroid = np.array(list(pred_region.centroid))
                distance = np.linalg.norm(pred_centroid - target_centroid)
                area_image = np.array(pred_region.area)
                if distance < self.match_distance:
                    distance_match.append(distance)
                    image_area_match.append(area_image)
                    del pred_regions[pred_index]
                    break
        dismatch = [area for area in image_area_total if area not in image_area_match]
        self.dismatch_pixel += float(np.sum(dismatch))
        self.all_pixel += float(target.shape[0] * target.shape[1])
        self.pd += len(distance_match)

    def compute(self) -> dict[str, float]:
        eps = np.spacing(1)
        pix_acc = self.total_correct / (self.total_label + eps)
        miou = self.total_inter / (self.total_union + eps)
        pd = self.pd / (self.targets + eps)
        fa = self.dismatch_pixel / (self.all_pixel + eps)
        return {
            "pixAcc": float(pix_acc),
            "mIoU": float(miou),
            "PD": float(pd),
            "FA": float(fa * 1e6),
            "FA_raw": float(fa),
            "FA_x1e6": float(fa * 1e6),
            "FA_x1e7": float(fa * 1e7),
        }
