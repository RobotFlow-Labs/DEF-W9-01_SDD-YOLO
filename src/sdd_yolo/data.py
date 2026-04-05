from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class Sample:
    image: torch.Tensor
    targets: torch.Tensor
    path: str
    original_size: tuple[int, int]


def parse_yolo_label_file(path: Path) -> torch.Tensor:
    if not path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)

    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if len(tokens) != 5:
                continue
            rows.append([float(v) for v in tokens])

    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)

    return torch.tensor(rows, dtype=torch.float32)


def _default_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        candidate = Path(*parts).with_suffix(".txt")
        return candidate
    return image_path.with_suffix(".txt")


def _load_image(image_path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, (original_size[1], original_size[0])


class YoloDetectionDataset(Dataset[Sample]):
    def __init__(self, root: str | Path, split_file: str | Path, image_size: int) -> None:
        self.root = Path(root)
        self.image_size = image_size

        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = self.root / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with split_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        self.image_paths = [self.root / line for line in lines]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Sample:
        image_path = self.image_paths[index]
        image_tensor, original_size = _load_image(image_path, self.image_size)
        targets = parse_yolo_label_file(_default_label_path(image_path))
        return Sample(
            image=image_tensor,
            targets=targets,
            path=str(image_path),
            original_size=original_size,
        )


class SyntheticDroneDataset(Dataset[Sample]):
    """Small synthetic dataset for local smoke tests when real data is absent."""

    def __init__(self, length: int = 32, image_size: int = 256, num_classes: int = 1) -> None:
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Sample:
        gen = torch.Generator().manual_seed(index)
        image = torch.rand((3, self.image_size, self.image_size), generator=gen)

        cx = torch.rand(1, generator=gen).item() * 0.8 + 0.1
        cy = torch.rand(1, generator=gen).item() * 0.8 + 0.1
        w = torch.rand(1, generator=gen).item() * 0.08 + 0.02
        h = torch.rand(1, generator=gen).item() * 0.08 + 0.02
        cls = float(torch.randint(0, self.num_classes, (1,), generator=gen).item())
        targets = torch.tensor([[cls, cx, cy, w, h]], dtype=torch.float32)

        return Sample(
            image=image,
            targets=targets,
            path=f"synthetic_{index:05d}.png",
            original_size=(self.image_size, self.image_size),
        )


def yolo_collate(batch: Iterable[Sample]) -> dict[str, torch.Tensor | list[str] | list[tuple[int, int]]]:
    images = []
    packed_targets = []
    paths: list[str] = []
    original_sizes: list[tuple[int, int]] = []

    for batch_idx, sample in enumerate(batch):
        images.append(sample.image)
        paths.append(sample.path)
        original_sizes.append(sample.original_size)

        if sample.targets.numel() > 0:
            idx_column = torch.full((sample.targets.shape[0], 1), batch_idx, dtype=torch.float32)
            packed_targets.append(torch.cat([idx_column, sample.targets], dim=1))

    target_tensor = (
        torch.cat(packed_targets, dim=0)
        if packed_targets
        else torch.zeros((0, 6), dtype=torch.float32)
    )

    return {
        "images": torch.stack(images, dim=0),
        "targets": target_tensor,
        "paths": paths,
        "original_sizes": original_sizes,
    }
