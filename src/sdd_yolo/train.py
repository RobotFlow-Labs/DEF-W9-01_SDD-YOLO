from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from sdd_yolo.config import RuntimeConfig, load_runtime_config
from sdd_yolo.data import SyntheticDroneDataset, YoloDetectionDataset, yolo_collate
from sdd_yolo.losses import DetectionLoss
from sdd_yolo.models.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _orthogonalize_gradients(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.grad is None or param.grad.ndim < 2:
            continue
        g = param.grad.data.view(param.grad.shape[0], -1)
        g = torch.nn.functional.normalize(g, dim=1)
        param.grad.data.copy_(g.view_as(param.grad.data))


def _build_dataloader(cfg: RuntimeConfig, split: str) -> DataLoader:
    split_file = cfg.data.train_split if split == "train" else cfg.data.val_split
    split_path = Path(cfg.data.root) / split_file
    if split_path.exists():
        dataset = YoloDetectionDataset(cfg.data.root, split_file, cfg.data.image_size)
    elif cfg.data.allow_synthetic:
        dataset = SyntheticDroneDataset(length=64, image_size=cfg.data.image_size, num_classes=cfg.data.num_classes)
    else:
        raise FileNotFoundError(f"Split file not found: {split_path}")

    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.train.workers,
        collate_fn=yolo_collate,
    )


def _checkpoint_path(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"epoch_{epoch:03d}.pt"


def train(cfg: RuntimeConfig, resume: str | None = None) -> dict[str, float | int | str]:
    set_seed(cfg.train.seed)

    device_name = cfg.train.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = build_model(cfg.model).to(device)
    if resume and Path(resume).exists():
        state = torch.load(resume, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    train_loader = _build_dataloader(cfg, split="train")
    criterion = DetectionLoss(cfg.data.num_classes, stal_gamma=cfg.train.stal_gamma)

    optimizer_name = cfg.train.optimizer.lower()
    if optimizer_name in {"sgd", "musgd"}:
        optimizer = optim.SGD(model.parameters(), lr=cfg.train.learning_rate, momentum=0.9, weight_decay=cfg.train.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    start = time.perf_counter()
    last_ckpt = ""
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in train_loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(images)
            losses = criterion(preds, targets)
            losses.total.backward()

            if optimizer_name == "musgd":
                _orthogonalize_gradients(model)

            optimizer.step()
            global_step += 1

            if cfg.train.max_steps > 0 and global_step >= cfg.train.max_steps:
                break

        if (epoch + 1) % cfg.train.save_every == 0:
            ckpt_path = _checkpoint_path(output_dir, epoch + 1)
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "step": global_step}, ckpt_path)
            last_ckpt = str(ckpt_path)

        if cfg.train.max_steps > 0 and global_step >= cfg.train.max_steps:
            break

    elapsed = max(1e-6, time.perf_counter() - start)
    return {
        "steps": global_step,
        "epochs": epoch + 1,
        "seconds": elapsed,
        "steps_per_second": global_step / elapsed,
        "checkpoint": last_ckpt,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SDD-YOLO training")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_runtime_config(args.config)

    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.output_dir:
        cfg.train.output_dir = args.output_dir

    metrics = train(cfg, resume=args.resume)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
