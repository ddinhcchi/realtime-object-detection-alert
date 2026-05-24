import time
from dataclasses import dataclass

import cv2
import numpy as np


def rect_polygon(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Axis-aligned rectangle polygon from two opposite corners.

    Returns the 4 vertices in clockwise order, dtype int32 — the shape OpenCV
    expects for `fillPoly`, `polylines` and `pointPolygonTest`.
    """
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.int32,
    )


def centered_polygon(frame_w: int, frame_h: int, padding_ratio: float) -> np.ndarray:
    """Centred rectangle inset by `padding_ratio` of frame size on each side."""
    pad_x = int(frame_w * padding_ratio)
    pad_y = int(frame_h * padding_ratio)
    return rect_polygon(pad_x, pad_y, frame_w - pad_x, frame_h - pad_y)


def polygon_from_points(points: list[tuple[int, int]]) -> np.ndarray:
    """Build an int32 polygon from a list of (x, y) points — for future
    free-form zones drawn in the UI."""
    if len(points) < 3:
        raise ValueError("polygon needs at least 3 points")
    return np.array(points, dtype=np.int32)


def point_in_polygon(point: tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


@dataclass
class DwellEvent:
    track_id: int
    class_name: str
    dwell_seconds: float


class DwellTracker:
    """Per-track-id timer. Fires once when a track stays in the zone for >= threshold.

    Re-fires only after the track has left the zone and `cooldown_seconds` elapsed.
    """

    def __init__(self, threshold_seconds: float, cooldown_seconds: float = 30):
        self.threshold = threshold_seconds
        self.cooldown = cooldown_seconds
        self._entered_at: dict[int, float] = {}
        self._last_alert_at: dict[int, float] = {}

    def update(
        self,
        tracks: list,
        polygon: np.ndarray,
        watch_class: str,
    ) -> list[DwellEvent]:
        now = time.time()
        events: list[DwellEvent] = []
        seen_in_zone: set[int] = set()

        for tr in tracks:
            if tr.class_name != watch_class or tr.track_id is None:
                continue
            cx = (tr.bbox[0] + tr.bbox[2]) // 2
            cy = (tr.bbox[1] + tr.bbox[3]) // 2
            if not point_in_polygon((cx, cy), polygon):
                continue

            seen_in_zone.add(tr.track_id)
            entered = self._entered_at.get(tr.track_id)
            if entered is None:
                self._entered_at[tr.track_id] = now
                continue

            dwell = now - entered
            if dwell < self.threshold:
                continue

            last_alert = self._last_alert_at.get(tr.track_id, 0)
            if now - last_alert < self.cooldown:
                continue

            events.append(
                DwellEvent(
                    track_id=tr.track_id,
                    class_name=tr.class_name,
                    dwell_seconds=dwell,
                )
            )
            self._last_alert_at[tr.track_id] = now

        # Reset timers for tracks that left the zone this frame
        for tid in list(self._entered_at):
            if tid not in seen_in_zone:
                self._entered_at.pop(tid, None)

        return events

    def dwell_for(self, track_id: int) -> float:
        entered = self._entered_at.get(track_id)
        return 0.0 if entered is None else time.time() - entered
