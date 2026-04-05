from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from sdd_yolo.config import RuntimeConfig, load_runtime_config
from sdd_yolo.models.model import build_model
from sdd_yolo.visualize import save_visualizations


def _load_image(path: Path, image_size: int) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB")
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return tensor, image


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in input_path.rglob("*") if p.suffix.lower() in exts])


def load_model(cfg: RuntimeConfig, checkpoint: str | None, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg.model).to(device)
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def run_inference(
    cfg: RuntimeConfig,
    input_path: Path,
    checkpoint: str | None,
    output_dir: Path,
    device: torch.device,
) -> dict[str, float | int]:
    model = load_model(cfg, checkpoint, device)
    paths = _collect_images(input_path)
    if not paths:
        raise FileNotFoundError(f"No images found in {input_path}")

    raw_images = []
    detections = []
    names = []
    json_rows = []

    start = time.perf_counter()
    for path in paths:
        tensor, raw = _load_image(path, cfg.data.image_size)
        tensor = tensor.unsqueeze(0).to(device)
        pred = model.predict(
            tensor,
            conf_threshold=cfg.infer.conf_threshold,
            iou_threshold=cfg.infer.iou_threshold,
            max_detections=cfg.infer.max_detections,
            nms_free=cfg.infer.nms_free,
        )[0].cpu()

        raw_images.append(raw)
        detections.append(pred)
        names.append(path.stem)
        json_rows.append(
            {
                "image": str(path),
                "detections": [
                    {
                        "x1": float(d[0]),
                        "y1": float(d[1]),
                        "x2": float(d[2]),
                        "y2": float(d[3]),
                        "score": float(d[4]),
                        "class": int(d[5]),
                    }
                    for d in pred
                ],
            }
        )

    elapsed = max(1e-6, time.perf_counter() - start)
    fps = len(paths) / elapsed

    output_dir.mkdir(parents=True, exist_ok=True)
    save_visualizations(raw_images, detections, output_dir, names)
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(json_rows, handle, indent=2)

    return {"num_images": len(paths), "fps": fps}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SDD-YOLO inference")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="artifacts/infer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--nms-free", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_runtime_config(args.config)
    if args.nms_free:
        cfg.infer.nms_free = True

    device_name = args.device or cfg.train.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    metrics = run_inference(
        cfg=cfg,
        input_path=Path(args.input),
        checkpoint=args.checkpoint,
        output_dir=Path(args.output),
        device=device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
