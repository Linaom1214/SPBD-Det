from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


class InfraredSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_dir: str,
        image_dir: str = "images",
        mask_dir: str = "masks",
        suffixes: tuple[str, ...] | list[str] = ("png", "jpg", "jpeg", "bmp", "tif", "tiff"),
        image_size: int = 512,
        crop_size: int = 512,
        normalize_mean: tuple[float, float, float] | list[float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] | list[float] = (0.229, 0.224, 0.225),
        training: bool = False,
        augment: bool = False,
        mask_binarize: bool = True,
        recursive_sequences: bool = False,
        split_file: str | Path | None = None,
    ):
        self.root = Path(root)
        self.split_dir = split_dir
        self.split_file = Path(split_file) if split_file else None
        self.split_root = self.root / split_dir
        self.images_dir = self.split_root / image_dir
        self.masks_dir = self.split_root / mask_dir
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        self.training = training
        self.augment = augment
        self.mask_binarize = mask_binarize
        self.suffixes = tuple(s.lower().lstrip(".") for s in suffixes)
        self.samples = []
        missing = []
        if self.split_file is not None:
            if not self.split_file.is_absolute():
                self.split_file = self.root / self.split_file
            if not self.split_file.is_file():
                raise FileNotFoundError(f"Split file not found: {self.split_file}")
            images_dir = self.root / image_dir
            masks_dir = self.root / mask_dir
            for line in self.split_file.read_text(encoding="utf-8").splitlines():
                name = line.strip()
                if not name:
                    continue
                image_path = self._resolve_indexed_path(images_dir, name)
                mask_path = self._resolve_indexed_path(masks_dir, name)
                if image_path is None or mask_path is None:
                    missing.append(name)
                else:
                    self.samples.append((image_path, mask_path, image_path.name))
        elif recursive_sequences:
            sequence_roots = sorted(
                p for p in self.split_root.iterdir()
                if p.is_dir() and (p / image_dir).is_dir() and (p / mask_dir).is_dir()
            )
            for sequence_root in sequence_roots:
                images_dir = sequence_root / image_dir
                masks_dir = sequence_root / mask_dir
                image_paths = sorted(
                    p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower().lstrip(".") in self.suffixes
                )
                for image_path in image_paths:
                    mask_path = self._find_mask_path(image_path, masks_dir)
                    if mask_path is None:
                        missing.append(str(image_path.relative_to(self.split_root)))
                    else:
                        self.samples.append((image_path, mask_path, str(image_path.relative_to(images_dir.parent))))
        else:
            if not self.images_dir.is_dir():
                raise FileNotFoundError(f"Image directory not found: {self.images_dir}")
            if not self.masks_dir.is_dir():
                raise FileNotFoundError(f"Mask directory not found: {self.masks_dir}")
            image_paths = sorted(
                p for p in self.images_dir.iterdir()
                if p.is_file() and p.suffix.lower().lstrip(".") in self.suffixes
            )
            for image_path in image_paths:
                mask_path = self._find_mask_path(image_path, self.masks_dir)
                if mask_path is None:
                    missing.append(image_path.name)
                else:
                    self.samples.append((image_path, mask_path, image_path.name))
        if not self.samples:
            raise RuntimeError(f"No images found in {self.split_root}")
        if missing:
            raise FileNotFoundError(f"Missing masks for {len(missing)} images, first: {missing[0]}")
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])

    def _find_mask_path(self, image_path: Path, masks_dir: Path) -> Path | None:
        exact = masks_dir / image_path.name
        if exact.is_file():
            return exact
        for suffix in self.suffixes:
            candidate = masks_dir / f"{image_path.stem}.{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _resolve_indexed_path(self, base_dir: Path, name: str) -> Path | None:
        raw = Path(name)
        candidates = []
        if raw.suffix:
            candidates.append(base_dir / raw)
        else:
            candidates.extend(base_dir / f"{name}.{suffix}" for suffix in self.suffixes)
        if "_" in name:
            prefix, frame = name.rsplit("_", 1)
            if frame.isdigit():
                candidates.extend(base_dir / prefix / f"{frame}.{suffix}" for suffix in self.suffixes)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path, name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = self._sync_transform(image, mask)
        image_tensor = self.image_transform(np.array(image))
        mask_array = np.array(mask, dtype=np.float32)
        if self.mask_binarize:
            mask_array = (mask_array > 127).astype(np.float32)
        else:
            mask_array = mask_array / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor, name

    def _sync_transform(self, image: Image.Image, mask: Image.Image):
        if self.training and self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            long_size = random.randint(int(self.image_size * 0.5), int(self.image_size * 2.0))
            w, h = image.size
            if h > w:
                oh = long_size
                ow = int(w * long_size / h + 0.5)
                short_size = ow
            else:
                ow = long_size
                oh = int(h * long_size / w + 0.5)
                short_size = oh
            image = image.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            if short_size < self.crop_size:
                pad_h = max(self.crop_size - oh, 0)
                pad_w = max(self.crop_size - ow, 0)
                image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)
            w, h = image.size
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
            image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            if random.random() < 0.5:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        else:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        return image, mask


def build_dataloader(cfg, split: str, training: bool, distributed: bool = False):
    data_cfg = cfg.data
    split_dir = data_cfg.train_dir if split == "train" else data_cfg.val_dir if split == "val" else data_cfg.test_dir
    dataset = InfraredSegmentationDataset(
        root=data_cfg.root,
        split_dir=split_dir,
        image_dir=data_cfg.image_dir,
        mask_dir=data_cfg.mask_dir,
        suffixes=data_cfg.suffixes,
        image_size=data_cfg.image_size,
        crop_size=data_cfg.crop_size,
        normalize_mean=data_cfg.normalize_mean,
        normalize_std=data_cfg.normalize_std,
        training=training,
        augment=bool(data_cfg.train_aug) and training,
        mask_binarize=bool(getattr(data_cfg, "mask_binarize", True)),
        recursive_sequences=bool(getattr(data_cfg, "recursive_sequences", False)),
        split_file=getattr(data_cfg, f"{split}_split_file", None),
    )
    sampler = DistributedSampler(dataset, shuffle=training) if distributed else None
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    return DataLoader(
        dataset,
        batch_size=data_cfg.train_batch_size if training else data_cfg.eval_batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        generator=generator if sampler is None else None,
    )
