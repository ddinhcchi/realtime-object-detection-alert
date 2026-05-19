"""Threaded RTSP reader: always exposes the *latest* frame, drops stale ones.

RTSP streams over OpenCV's internal buffer can lag behind real-time by several
seconds. This reader runs a background thread that drains frames as fast as the
camera produces them and keeps only the most recent one.
"""
import os
import threading
import time

import cv2
import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class RtspReader:
    def __init__(self, url: str, reconnect_seconds: float = 2.0):
        self.url = url
        self.reconnect_seconds = reconnect_seconds
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        cap = self._open()
        while not self._stop.is_set():
            if not cap.isOpened():
                self._connected = False
                cap.release()
                time.sleep(self.reconnect_seconds)
                cap = self._open()
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                self._connected = False
                cap.release()
                time.sleep(self.reconnect_seconds)
                cap = self._open()
                continue
            self._connected = True
            with self._lock:
                self._frame = frame
        cap.release()
