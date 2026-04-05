from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any, Dict


@dataclass(slots=True)
class DataConfig:
    root: str = "data"
    train_split: str = "train.txt"
    val_split: str = "val.txt"
    test_split: str = "test.txt"
    image_size: int = 1024
    num_classes: int = 1
    allow_synthetic: bool = True


@dataclass(slots=True)
class ModelConfig:
    name: str = "sdd_yolo_n"
    width_mult: float = 0.5
    depth_mult: float = 0.5
    num_classes: int = 1
    use_p2_head: bool = True
    use_dual_attention: bool = True
    dfl_enabled: bool = False
    nms_free: bool = True


@dataclass(slots=True)
class DistillConfig:
    enabled: bool = False
    lambda_weight: float = 0.5
    temperature: float = 3.0


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 8
    epochs: int = 100
    max_steps: int = -1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 2
    device: str = "cpu"
    seed: int = 42
    save_every: int = 1
    output_dir: str = "artifacts"
    optimizer: str = "musgd"
    stal_gamma: float = 2.0


@dataclass(slots=True)
class InferConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    max_detections: int = 300
    nms_free: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)


def _to_dataclass(instance: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def runtime_from_dict(payload: Dict[str, Any]) -> RuntimeConfig:
    cfg = RuntimeConfig()
    if "data" in payload:
        cfg.data = _to_dataclass(cfg.data, payload["data"])
    if "model" in payload:
        cfg.model = _to_dataclass(cfg.model, payload["model"])
    if "train" in payload:
        cfg.train = _to_dataclass(cfg.train, payload["train"])
    if "infer" in payload:
        cfg.infer = _to_dataclass(cfg.infer, payload["infer"])
    if "distill" in payload:
        cfg.distill = _to_dataclass(cfg.distill, payload["distill"])

    cfg.model.num_classes = cfg.data.num_classes
    return cfg


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    return runtime_from_dict(payload)


def merge_runtime_config(base: RuntimeConfig, patch: Dict[str, Any]) -> RuntimeConfig:
    payload = {
        "data": vars(base.data),
        "model": vars(base.model),
        "train": vars(base.train),
        "infer": vars(base.infer),
        "distill": vars(base.distill),
    }
    for section, values in patch.items():
        if section in payload and isinstance(values, dict):
            payload[section].update(values)
    return runtime_from_dict(payload)
