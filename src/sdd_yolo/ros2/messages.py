from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass(slots=True)
class DetectionFrame:
    frame_id: str
    timestamp_ns: int
    detections: list[Detection]
