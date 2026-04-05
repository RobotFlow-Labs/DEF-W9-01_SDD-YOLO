from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from sdd_yolo.config import load_runtime_config
from sdd_yolo.data import YoloDetectionDataset, yolo_collate


def _write_sample_dataset(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)

    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    image_path = root / "images" / "sample.png"
    image.save(image_path)

    label_path = root / "labels" / "sample.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    (root / "train.txt").write_text("images/sample.png\n", encoding="utf-8")


def test_load_default_config() -> None:
    cfg = load_runtime_config("configs/default.toml")
    assert cfg.model.name == "sdd_yolo_n"
    assert cfg.model.num_classes == cfg.data.num_classes


def test_dataset_and_collate(tmp_path: Path) -> None:
    _write_sample_dataset(tmp_path)
    dataset = YoloDetectionDataset(tmp_path, "train.txt", image_size=128)
    sample = dataset[0]
    assert sample.image.shape == (3, 128, 128)
    assert sample.targets.shape[1] == 5

    batch = yolo_collate([sample, sample])
    assert batch["images"].shape[0] == 2
    assert batch["targets"].shape[1] == 6
