from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sdd_yolo.config import load_runtime_config
from sdd_yolo.infer import load_model


def export_onnx(config_path: str, checkpoint: str | None, output_path: str, opset: int = 17) -> Path:
    cfg = load_runtime_config(config_path)
    device = torch.device("cpu")
    model = load_model(cfg, checkpoint, device)
    model.eval()

    dummy = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["images"],
        output_names=["p2", "p3", "p4", "p5"],
        dynamic_axes={"images": {0: "batch"}},
        opset_version=opset,
    )
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export SDD-YOLO to ONNX")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="artifacts/export/sdd_yolo.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    exported = export_onnx(args.config, args.checkpoint, args.output, args.opset)
    print(f"Exported: {exported}")


if __name__ == "__main__":
    main()
