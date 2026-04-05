from __future__ import annotations

import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    half_w = w / 2.0
    half_h = h / 2.0
    return torch.stack((x - half_w, y - half_h, x + half_w, y + half_h), dim=-1)


def xyxy_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    tl = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    br = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_wh = (br - tl).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        iou = xyxy_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def clamp_boxes_xyxy(boxes: torch.Tensor, min_v: float = 0.0, max_v: float = 1.0) -> torch.Tensor:
    return boxes.clamp(min=min_v, max=max_v)
