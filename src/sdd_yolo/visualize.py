from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
import torch


def draw_detections(image: Image.Image, detections: torch.Tensor, color: str = "lime") -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    for det in detections:
        x1, y1, x2, y2, score, cls_idx = det.tolist()
        box = [x1 * width, y1 * height, x2 * width, y2 * height]
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], max(0, box[1] - 12)), f"{int(cls_idx)}:{score:.2f}", fill=color)

    return canvas


def save_visualizations(
    images: Iterable[Image.Image],
    detections: Iterable[torch.Tensor],
    output_dir: str | Path,
    base_names: Iterable[str],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for image, det, name in zip(images, detections, base_names):
        vis = draw_detections(image, det)
        vis.save(out / f"{name}.pred.png")
