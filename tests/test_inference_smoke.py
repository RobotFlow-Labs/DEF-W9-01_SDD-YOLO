from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from sdd_yolo.config import load_runtime_config
from sdd_yolo.infer import run_inference


def test_inference_smoke(tmp_path: Path) -> None:
    image = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    image_path = tmp_path / "input.png"
    image.save(image_path)

    cfg = load_runtime_config("configs/debug.toml")
    output = tmp_path / "out"
    metrics = run_inference(cfg, image_path, checkpoint=None, output_dir=output, device=torch.device("cpu"))

    assert metrics["num_images"] == 1
    assert (output / "predictions.json").exists()
