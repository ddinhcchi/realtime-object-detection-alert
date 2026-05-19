import io
import time
from pathlib import Path

import cv2
import numpy as np
import requests

TELEGRAM_API = "https://api.telegram.org"


class TelegramAlerter:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        cooldown_seconds: int = 10,
        snapshot_dir: str = "snapshots",
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown_seconds = cooldown_seconds
        self._last_sent: dict[str, float] = {}
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token) and bool(self.chat_id)

    def _can_send(self, key: str) -> bool:
        now = time.time()
        last = self._last_sent.get(key, 0)
        if now - last < self.cooldown_seconds:
            return False
        self._last_sent[key] = now
        return True

    def send(self, class_name: str, confidence: float, frame_bgr: np.ndarray) -> bool:
        if not self.is_configured:
            return False
        if not self._can_send(class_name):
            return False

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        snapshot_path = self.snapshot_dir / f"{class_name}-{timestamp}.jpg"
        cv2.imwrite(str(snapshot_path), frame_bgr)

        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return False

        caption = f"🚨 Detected: *{class_name}* (conf {confidence:.2f}) at {timestamp}"
        try:
            resp = requests.post(
                f"{TELEGRAM_API}/bot{self.bot_token}/sendPhoto",
                data={
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": "Markdown",
                },
                files={"photo": ("snapshot.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
                timeout=5,
            )
            return resp.ok
        except requests.RequestException:
            return False
