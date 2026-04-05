from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sdd_yolo.config import ModelConfig
from sdd_yolo.models.attention import DualAttention
from sdd_yolo.models.blocks import ConvBNAct, DownsampleStage, NeckFusion, _make_divisible
from sdd_yolo.models.head import DetectionHead


@dataclass(slots=True)
class ModelOutputs:
    p2: torch.Tensor
    p3: torch.Tensor
    p4: torch.Tensor
    p5: torch.Tensor


class SDDYOLO(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        base = _make_divisible(64 * cfg.width_mult)
        c2, c3, c4, c5 = base, base * 2, base * 4, base * 8

        self.stem = ConvBNAct(3, base, kernel=3, stride=2)
        self.stage2 = DownsampleStage(base, c2, depth=max(1, int(2 * cfg.depth_mult)))
        self.stage3 = DownsampleStage(c2, c3, depth=max(1, int(2 * cfg.depth_mult)))
        self.stage4 = DownsampleStage(c3, c4, depth=max(1, int(2 * cfg.depth_mult)))
        self.stage5 = DownsampleStage(c4, c5, depth=max(1, int(1 * cfg.depth_mult)))

        self.fuse4 = NeckFusion(c5, c4, c4)
        self.fuse3 = NeckFusion(c4, c3, c3)
        self.fuse2 = NeckFusion(c3, c2, c2)

        self.use_dual_attention = cfg.use_dual_attention
        if cfg.use_dual_attention:
            self.attn2 = DualAttention(c2)
            self.attn3 = DualAttention(c3)
        else:
            self.attn2 = nn.Identity()
            self.attn3 = nn.Identity()

        channels = [c2, c3, c4, c5] if cfg.use_p2_head else [c3, c4, c5]
        self.use_p2_head = cfg.use_p2_head
        self.head = DetectionHead(channels=channels, num_classes=cfg.num_classes)

    def forward_features(self, x: torch.Tensor) -> ModelOutputs:
        x = self.stem(x)
        p2_backbone = self.stage2(x)
        p3_backbone = self.stage3(p2_backbone)
        p4_backbone = self.stage4(p3_backbone)
        p5_backbone = self.stage5(p4_backbone)

        p4 = self.fuse4(p5_backbone, p4_backbone)
        p3 = self.fuse3(p4, p3_backbone)
        p2 = self.fuse2(p3, p2_backbone)

        p2 = self.attn2(p2)
        p3 = self.attn3(p3)
        return ModelOutputs(p2=p2, p3=p3, p4=p4, p5=p5_backbone)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.forward_features(x)
        if self.use_p2_head:
            head_inputs = [feats.p2, feats.p3, feats.p4, feats.p5]
        else:
            head_inputs = [feats.p3, feats.p4, feats.p5]
        return self.head(head_inputs)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        max_detections: int = 300,
        nms_free: bool = True,
    ) -> list[torch.Tensor]:
        self.eval()
        preds = self.forward(x)
        return self.head.decode(preds, conf_threshold, iou_threshold, max_detections, nms_free)


def build_model(cfg: ModelConfig) -> SDDYOLO:
    return SDDYOLO(cfg)


__all__ = ["SDDYOLO", "build_model"]
