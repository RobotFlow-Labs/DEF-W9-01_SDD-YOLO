from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    box: torch.Tensor
    obj: torch.Tensor
    cls: torch.Tensor


def _decode_cell(logit: torch.Tensor, gx: int, gy: int, w: int, h: int) -> torch.Tensor:
    px = (torch.sigmoid(logit[0]) + gx) / w
    py = (torch.sigmoid(logit[1]) + gy) / h
    pw = torch.exp(torch.clamp(logit[2], max=4.0)) / w
    ph = torch.exp(torch.clamp(logit[3], max=4.0)) / h
    return torch.stack([px, py, pw, ph])


class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int, stal_gamma: float = 2.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.stal_gamma = stal_gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, preds: list[torch.Tensor], targets: torch.Tensor) -> LossBreakdown:
        device = preds[0].device
        batch_size = preds[0].shape[0]
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)

        positive_count = 0

        per_scale_obj_targets = [torch.zeros_like(pred[:, 4]) for pred in preds]

        if targets.numel() > 0:
            for row in targets:
                b = int(row[0].item())
                cls = int(row[1].item())
                cx, cy, bw, bh = [float(v) for v in row[2:6]]

                area = max(bw * bh, 1e-6)
                if area <= 0.01:
                    scale_idx = 0
                elif area <= 0.03:
                    scale_idx = min(1, len(preds) - 1)
                elif area <= 0.09:
                    scale_idx = min(2, len(preds) - 1)
                else:
                    scale_idx = len(preds) - 1

                pred_map = preds[scale_idx]
                _, _, h, w = pred_map.shape
                gx = min(w - 1, max(0, int(cx * w)))
                gy = min(h - 1, max(0, int(cy * h)))

                per_scale_obj_targets[scale_idx][b, gy, gx] = 1.0
                pred_vec = pred_map[b, :, gy, gx]
                pred_box = _decode_cell(pred_vec, gx, gy, w, h)
                tgt_box = torch.tensor([cx, cy, bw, bh], device=device)

                small_weight = 1.0 + self.stal_gamma * (1.0 - math.sqrt(area))
                box_loss = box_loss + (self.l1(pred_box, tgt_box).mean() * small_weight)
                if self.num_classes > 0:
                    cls_target = torch.zeros((self.num_classes,), device=device)
                    if 0 <= cls < self.num_classes:
                        cls_target[cls] = 1.0
                    cls_loss = cls_loss + self.bce(pred_vec[5 : 5 + self.num_classes], cls_target).mean()

                positive_count += 1

        for scale_idx, pred_map in enumerate(preds):
            obj_target = per_scale_obj_targets[scale_idx]
            obj_loss = obj_loss + self.bce(pred_map[:, 4], obj_target).mean()

        normalizer = max(1, positive_count)
        box_loss = box_loss / normalizer
        cls_loss = cls_loss / normalizer
        total = box_loss + obj_loss + cls_loss

        return LossBreakdown(total=total, box=box_loss, obj=obj_loss, cls=cls_loss)
