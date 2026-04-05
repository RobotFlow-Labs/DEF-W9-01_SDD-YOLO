from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from sdd_yolo.config import load_runtime_config
from sdd_yolo.serve import create_app


def _make_png_bytes() -> bytes:
    image = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    buff = BytesIO()
    image.save(buff, format="PNG")
    return buff.getvalue()


def test_health_and_predict() -> None:
    cfg = load_runtime_config("configs/debug.toml")
    app = create_app(cfg, checkpoint=None)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    payload = _make_png_bytes()
    resp = client.post("/predict", files={"file": ("sample.png", payload, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "num_detections" in body
    assert "detections" in body
