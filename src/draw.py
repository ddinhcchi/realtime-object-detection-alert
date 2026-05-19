import cv2
import numpy as np

from .detector import Track

_COLOR_TRACK = (0, 255, 0)
_COLOR_ZONE = (0, 165, 255)
_COLOR_ZONE_HIT = (0, 0, 255)


def draw_zone(frame: np.ndarray, polygon: np.ndarray, hit: bool = False) -> None:
    color = _COLOR_ZONE_HIT if hit else _COLOR_ZONE
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, dst=frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    dwell_lookup,
) -> None:
    for tr in tracks:
        x1, y1, x2, y2 = tr.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR_TRACK, 2)
        label = f"#{tr.track_id} {tr.class_name} {tr.confidence:.2f}"
        if tr.track_id is not None:
            dwell = dwell_lookup(tr.track_id)
            if dwell > 0:
                label += f" | {dwell:.1f}s"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _COLOR_TRACK,
            2,
            cv2.LINE_AA,
        )


def draw_hud(frame: np.ndarray, device: str, latency_ms: float, n_tracks: int) -> None:
    overlay = f"{device.upper()} | {latency_ms:.1f} ms | tracks: {n_tracks}"
    cv2.rectangle(frame, (0, 0), (340, 32), (0, 0, 0), -1)
    cv2.putText(
        frame,
        overlay,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
