from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sdd_yolo.ros2.messages import Detection, DetectionFrame


@dataclass(slots=True)
class NodeConfig:
    input_topic: str = "/camera/rgb/image"
    output_topic: str = "/defense/sdd_yolo/detections"


class SDDYoloNode:
    """ROS2 adapter stub for ANIMA integration.

    The concrete rclpy publisher/subscriber wiring is intentionally deferred to
    server-side integration where ROS2 runtime is available.
    """

    def __init__(self, config: NodeConfig | None = None) -> None:
        self.config = config or NodeConfig()

    def process(self, frame_id: str, timestamp_ns: int, raw_detections: Iterable[tuple[float, float, float, float, float, int]]) -> DetectionFrame:
        detections = [
            Detection(x1=d[0], y1=d[1], x2=d[2], y2=d[3], score=d[4], class_id=d[5])
            for d in raw_detections
        ]
        return DetectionFrame(frame_id=frame_id, timestamp_ns=timestamp_ns, detections=detections)
