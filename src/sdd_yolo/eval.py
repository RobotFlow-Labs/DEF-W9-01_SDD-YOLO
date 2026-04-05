from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from sdd_yolo.config import RuntimeConfig, load_runtime_config
from sdd_yolo.data import SyntheticDroneDataset, YoloDetectionDataset, yolo_collate
from sdd_yolo.infer import load_model
from sdd_yolo.ops import xywh_to_xyxy, xyxy_iou


def _collect_ground_truth(batch_targets: torch.Tensor) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = {}
    for row in batch_targets:
        b = int(row[0].item())
        grouped.setdefault(b, []).append(row[1:6])
    out: dict[int, torch.Tensor] = {}
    for k, rows in grouped.items():
        out[k] = torch.stack(rows, dim=0)
    return out


def _average_precision(tp: torch.Tensor, fp: torch.Tensor, num_gt: int) -> float:
    if num_gt == 0:
        return 0.0
    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    recall = tp_cum / max(1, num_gt)

    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
    return float(ap)


def evaluate(cfg: RuntimeConfig, checkpoint: str | None, device: torch.device) -> dict[str, float]:
    split_path = Path(cfg.data.root) / cfg.data.val_split
    if split_path.exists():
        dataset = YoloDetectionDataset(cfg.data.root, cfg.data.val_split, cfg.data.image_size)
    elif cfg.data.allow_synthetic:
        dataset = SyntheticDroneDataset(length=24, image_size=cfg.data.image_size, num_classes=cfg.data.num_classes)
    else:
        raise FileNotFoundError("Validation split not found and synthetic data disabled.")

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=yolo_collate)
    model = load_model(cfg, checkpoint, device)

    thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = {thr: [] for thr in thresholds}

    total_gt = 0
    start = time.perf_counter()

    for batch in loader:
        images = batch["images"].to(device)
        preds = model.predict(
            images,
            conf_threshold=cfg.infer.conf_threshold,
            iou_threshold=cfg.infer.iou_threshold,
            max_detections=cfg.infer.max_detections,
            nms_free=cfg.infer.nms_free,
        )
        gt_grouped = _collect_ground_truth(batch["targets"])

        for idx, pred in enumerate(preds):
            gt = gt_grouped.get(idx, torch.zeros((0, 5)))
            total_gt += int(gt.shape[0])
            if gt.numel() == 0:
                continue

            gt_xyxy = xywh_to_xyxy(gt[:, 1:5])
            pred = pred.cpu()
            if pred.numel() == 0:
                continue

            order = pred[:, 4].argsort(descending=True)
            pred = pred[order]
            ious = xyxy_iou(pred[:, :4], gt_xyxy)

            for thr in thresholds:
                assigned = torch.zeros((gt_xyxy.shape[0],), dtype=torch.bool)
                tp = torch.zeros((pred.shape[0],), dtype=torch.float32)
                fp = torch.zeros((pred.shape[0],), dtype=torch.float32)
                for p_idx in range(pred.shape[0]):
                    max_iou, gt_idx = ious[p_idx].max(dim=0)
                    if max_iou >= thr and not assigned[gt_idx]:
                        assigned[gt_idx] = True
                        tp[p_idx] = 1.0
                    else:
                        fp[p_idx] = 1.0
                aps[thr].append(_average_precision(tp, fp, gt_xyxy.shape[0]))

    elapsed = max(1e-6, time.perf_counter() - start)
    fps = len(dataset) / elapsed

    map50 = float(sum(aps[0.5]) / max(1, len(aps[0.5])))
    map5095 = float(sum(sum(v) / max(1, len(v)) for v in aps.values()) / len(aps))

    return {
        "map_50": map50,
        "map_50_95": map5095,
        "num_images": float(len(dataset)),
        "num_gt": float(total_gt),
        "fps": fps,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SDD-YOLO evaluation")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="artifacts/eval/report.json")
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_runtime_config(args.config)

    device_name = args.device or cfg.train.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    metrics = evaluate(cfg, args.checkpoint, device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
