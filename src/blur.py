"""Face blur for privacy in demo recordings.

Uses YuNet (OpenCV Zoo) — an open-source ONNX face detector that ships ~340KB
weights and runs on CPU at real-time. Weights auto-download on first use.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class FaceBlurrer:
    def __init__(self, weights_dir: str = "weights", score_threshold: float = 0.6):
        self.weights_path = Path(weights_dir) / "yunet_2023mar.onnx"
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.weights_path.exists():
            urllib.request.urlretrieve(YUNET_URL, self.weights_path)
        self.detector = cv2.FaceDetectorYN.create(
            str(self.weights_path),
            "",
            (320, 320),
            score_threshold,
            0.3,
            5000,
        )
        self._input_size: tuple[int, int] | None = None

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self._input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)

        _, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return frame_bgr

        out = frame_bgr.copy()
        for face in faces:
            x, y, bw, bh = face[:4].astype(int)
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            if bw <= 0 or bh <= 0:
                continue
            roi = out[y : y + bh, x : x + bw]
            out[y : y + bh, x : x + bw] = cv2.GaussianBlur(
                roi, (0, 0), sigmaX=15, sigmaY=15
            )
        return out
