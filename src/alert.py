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
        snapshot_dir: str = "snapshots",
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token) and bool(self.chat_id)

    def send_dwell(
        self,
        track_id: int,
        class_name: str,
        dwell_seconds: float,
        frame_bgr: np.ndarray,
    ) -> bool:
        if not self.is_configured:
            return False

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        slug = f"{class_name}-id{track_id}-{int(time.time())}"
        snapshot_path = self.snapshot_dir / f"{slug}.jpg"
        cv2.imwrite(str(snapshot_path), frame_bgr)

        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return False

        caption = (
            f"🚨 *Zone intrusion*\n"
            f"Class: `{class_name}`  Track: `#{track_id}`\n"
            f"Dwell: *{dwell_seconds:.1f}s*\n"
            f"Time: {timestamp}"
        )
        try:
            resp = requests.post(
                f"{TELEGRAM_API}/bot{self.bot_token}/sendPhoto",
                data={
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": "Markdown",
                },
                files={
                    "photo": ("snapshot.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")
                },
                timeout=5,
            )
            return resp.ok
        except requests.RequestException:
            return False
