from __future__ import annotations

import torch
from torch import nn


def _make_divisible(v: float, divisor: int = 8) -> int:
    return int((v + divisor / 2) // divisor * divisor)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = max(8, int(channels * expansion))
        self.cv1 = ConvBNAct(channels, hidden, kernel=1)
        self.cv2 = ConvBNAct(hidden, channels, kernel=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class CSPBlock(nn.Module):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[Bottleneck(channels) for _ in range(max(1, depth))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        self.down = ConvBNAct(in_channels, out_channels, kernel=3, stride=2)
        self.csp = CSPBlock(out_channels, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.csp(self.down(x))


class NeckFusion(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = ConvBNAct(in_channels, out_channels, kernel=1)
        self.merge = ConvBNAct(out_channels + skip_channels, out_channels, kernel=3)

    def forward(self, top: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        top = self.reduce(top)
        top = nn.functional.interpolate(top, size=skip.shape[-2:], mode="nearest")
        return self.merge(torch.cat([top, skip], dim=1))


__all__ = [
    "_make_divisible",
    "ConvBNAct",
    "CSPBlock",
    "DownsampleStage",
    "NeckFusion",
]
