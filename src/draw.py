import cv2
import numpy as np

from .detector import Detection

_COLOR_HIT = (0, 0, 255)
_COLOR_NORMAL = (0, 255, 0)


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    watch_classes: set[str],
    latency_ms: float,
    device: str,
) -> np.ndarray:
    out = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = _COLOR_HIT if det.class_name in watch_classes else _COLOR_NORMAL
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            out, label, (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    overlay = f"{device.upper()} | {latency_ms:.1f} ms | {len(detections)} obj"
    cv2.putText(
        out, overlay, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out
