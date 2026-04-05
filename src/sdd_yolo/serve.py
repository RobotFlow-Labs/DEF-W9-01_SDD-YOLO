from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
import uvicorn

from sdd_yolo.config import RuntimeConfig, load_runtime_config
from sdd_yolo.infer import load_model


class ServiceState:
    def __init__(self, cfg: RuntimeConfig, checkpoint: str | None) -> None:
        device_name = cfg.train.device
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.cfg = cfg
        self.model = load_model(cfg, checkpoint, self.device)


def _image_to_tensor(raw: bytes, image_size: int) -> torch.Tensor:
    image = Image.open(BytesIO(raw)).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def create_app(cfg: RuntimeConfig, checkpoint: str | None) -> FastAPI:
    state = ServiceState(cfg, checkpoint)
    app = FastAPI(title="DEF-sdd-yolo", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, object]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        raw = await file.read()
        try:
            tensor = _image_to_tensor(raw, state.cfg.data.image_size).to(state.device)
        except Exception as exc:  # pragma: no cover - guarded by tests
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        detections = state.model.predict(
            tensor,
            conf_threshold=state.cfg.infer.conf_threshold,
            iou_threshold=state.cfg.infer.iou_threshold,
            max_detections=state.cfg.infer.max_detections,
            nms_free=state.cfg.infer.nms_free,
        )[0].cpu()

        return {
            "num_detections": int(detections.shape[0]),
            "detections": [
                {
                    "x1": float(row[0]),
                    "y1": float(row[1]),
                    "x2": float(row[2]),
                    "y2": float(row[3]),
                    "score": float(row[4]),
                    "class": int(row[5]),
                }
                for row in detections
            ],
        }

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DEF-sdd-yolo FastAPI service")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_runtime_config(args.config)
    app = create_app(cfg, args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
