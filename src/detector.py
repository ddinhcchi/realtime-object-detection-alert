import time
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


def resolve_device(preferred: str) -> str:
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Detector:
    def __init__(self, weights: str = "yolov8n.pt", device: str = "mps"):
        self.device = resolve_device(device)
        self.model = YOLO(weights)
        self.class_names: dict[int, str] = self.model.names
        self.last_latency_ms: float = 0.0

    def predict(
        self, frame_bgr: np.ndarray, conf: float = 0.5, classes: list[int] | None = None
    ) -> list[Detection]:
        t0 = time.perf_counter()
        results = self.model.predict(
            frame_bgr,
            device=self.device,
            conf=conf,
            classes=classes,
            verbose=False,
        )
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None or boxes.cls is None:
            return detections
        for cls_t, conf_t, xyxy_t in zip(boxes.cls, boxes.conf, boxes.xyxy):
            cid = int(cls_t.item())
            x1, y1, x2, y2 = (int(v) for v in xyxy_t.tolist())
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=self.class_names.get(cid, str(cid)),
                    confidence=float(conf_t.item()),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections

    def class_name_to_id(self, name: str) -> int | None:
        for cid, cname in self.class_names.items():
            if cname == name:
                return cid
        return None
