from __future__ import annotations

import torch

from sdd_yolo.config import RuntimeConfig
from sdd_yolo.losses import DetectionLoss
from sdd_yolo.models.model import build_model


def test_model_forward_and_loss() -> None:
    cfg = RuntimeConfig()
    cfg.data.image_size = 128
    cfg.model.width_mult = 0.25
    cfg.model.depth_mult = 0.25

    model = build_model(cfg.model)
    images = torch.rand(2, 3, 128, 128)
    preds = model(images)

    assert len(preds) >= 3
    for p in preds:
        assert p.shape[0] == 2
        assert p.shape[1] == 5 + cfg.data.num_classes

    targets = torch.tensor(
        [
            [0.0, 0.0, 0.5, 0.5, 0.08, 0.08],
            [1.0, 0.0, 0.4, 0.6, 0.06, 0.06],
        ],
        dtype=torch.float32,
    )
    criterion = DetectionLoss(num_classes=cfg.data.num_classes)
    loss = criterion(preds, targets)
    assert torch.isfinite(loss.total)

    loss.total.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
