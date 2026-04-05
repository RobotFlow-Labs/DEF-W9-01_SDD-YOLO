from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from sdd_yolo.ops import clamp_boxes_xyxy, nms, xywh_to_xyxy


class DetectionHead(nn.Module):
    def __init__(self, channels: Iterable[int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        out_dim = 5 + num_classes
        self.pred_layers = nn.ModuleList([nn.Conv2d(c, out_dim, kernel_size=1) for c in channels])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [layer(feat) for layer, feat in zip(self.pred_layers, feats)]

    @torch.no_grad()
    def decode(
        self,
        preds: list[torch.Tensor],
        conf_threshold: float,
        iou_threshold: float,
        max_detections: int,
        nms_free: bool,
    ) -> list[torch.Tensor]:
        batch_size = preds[0].shape[0]
        outputs: list[torch.Tensor] = []

        for b in range(batch_size):
            detections: list[torch.Tensor] = []
            for pred in preds:
                det = self._decode_single_scale(pred[b], conf_threshold)
                if det.numel() > 0:
                    detections.append(det)

            if detections:
                merged = torch.cat(detections, dim=0)
                if not nms_free:
                    keep = nms(merged[:, :4], merged[:, 4], iou_threshold)
                    merged = merged[keep]
                if merged.shape[0] > max_detections:
                    merged = merged[merged[:, 4].argsort(descending=True)[:max_detections]]
            else:
                merged = torch.zeros((0, 6), device=preds[0].device)

            outputs.append(merged)

        return outputs

    def _decode_single_scale(self, pred: torch.Tensor, conf_threshold: float) -> torch.Tensor:
        _, height, width = pred.shape
        tx, ty, tw, th, tobj = pred[0], pred[1], pred[2], pred[3], pred[4]
        cls_logits = pred[5:]

        yv, xv = torch.meshgrid(
            torch.arange(height, device=pred.device),
            torch.arange(width, device=pred.device),
            indexing="ij",
        )

        cx = (torch.sigmoid(tx) + xv) / width
        cy = (torch.sigmoid(ty) + yv) / height
        bw = torch.exp(torch.clamp(tw, max=4.0)) / width
        bh = torch.exp(torch.clamp(th, max=4.0)) / height
        obj = torch.sigmoid(tobj)

        if cls_logits.shape[0] == 0:
            cls_prob = torch.ones_like(obj)
            cls_idx = torch.zeros_like(obj, dtype=torch.long)
        else:
            cls_prob_all = torch.sigmoid(cls_logits)
            cls_prob, cls_idx = cls_prob_all.max(dim=0)

        score = obj * cls_prob
        keep = score >= conf_threshold
        if keep.sum().item() == 0:
            return torch.zeros((0, 6), device=pred.device)

        boxes_xywh = torch.stack([cx[keep], cy[keep], bw[keep], bh[keep]], dim=1)
        boxes_xyxy = clamp_boxes_xyxy(xywh_to_xyxy(boxes_xywh))
        scores = score[keep].unsqueeze(1)
        classes = cls_idx[keep].float().unsqueeze(1)
        return torch.cat([boxes_xyxy, scores, classes], dim=1)


__all__ = ["DetectionHead"]
