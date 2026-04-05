from __future__ import annotations

import torch

from sdd_yolo.config import load_runtime_config
from sdd_yolo.eval import _average_precision, evaluate


def test_average_precision_sane() -> None:
    tp = torch.tensor([1.0, 0.0, 1.0])
    fp = torch.tensor([0.0, 1.0, 0.0])
    ap = _average_precision(tp, fp, num_gt=2)
    assert 0.0 <= ap <= 1.0


def test_eval_synthetic_runs() -> None:
    cfg = load_runtime_config("configs/debug.toml")
    cfg.data.allow_synthetic = True
    cfg.train.device = "cpu"
    metrics = evaluate(cfg, checkpoint=None, device=torch.device("cpu"))
    assert "map_50" in metrics
    assert "map_50_95" in metrics
    assert metrics["num_images"] > 0
