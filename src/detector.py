import time
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Track:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: int | None


def resolve_device(preferred: str) -> str:
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Detector:
    """YOLOv8 + ByteTrack tracker. One persistent model with stateful tracking."""

    def __init__(self, weights: str = "yolov8n.pt", device: str = "mps"):
        self.device = resolve_device(device)
        self.model = YOLO(weights)
        self.class_names: dict[int, str] = self.model.names
        self.last_latency_ms: float = 0.0

    def track(
        self,
        frame_bgr: np.ndarray,
        conf: float = 0.4,
        classes: list[int] | None = None,
    ) -> list[Track]:
        t0 = time.perf_counter()
        results = self.model.track(
            frame_bgr,
            device=self.device,
            conf=conf,
            classes=classes,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        out: list[Track] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or boxes.cls is None:
            return out

        track_ids = (
            boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes.cls)
        )
        for cls_t, conf_t, xyxy_t, tid in zip(boxes.cls, boxes.conf, boxes.xyxy, track_ids):
            cid = int(cls_t.item())
            x1, y1, x2, y2 = (int(v) for v in xyxy_t.tolist())
            out.append(
                Track(
                    class_id=cid,
                    class_name=self.class_names.get(cid, str(cid)),
                    confidence=float(conf_t.item()),
                    bbox=(x1, y1, x2, y2),
                    track_id=tid,
                )
            )
        return out

    def class_name_to_id(self, name: str) -> int | None:
        for cid, cname in self.class_names.items():
            if cname == name:
                return cid
        return None

    def reset_tracker(self) -> None:
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            trackers = getattr(self.model.predictor, "trackers", None)
            if trackers:
                for t in trackers:
                    if hasattr(t, "reset"):
                        t.reset()
