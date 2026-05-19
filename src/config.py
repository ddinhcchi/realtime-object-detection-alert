import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    rtsp_url: str = os.getenv("RTSP_URL", "")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    device: str = os.getenv("DEVICE", "mps")
    model_weights: str = os.getenv("MODEL_WEIGHTS", "yolov8n.pt")
    dwell_threshold_seconds: float = float(os.getenv("DWELL_THRESHOLD_SECONDS", "10"))
    alert_cooldown_seconds: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))
    watch_class: str = os.getenv("WATCH_CLASS", "person")
    enable_face_blur: bool = _env_bool("ENABLE_FACE_BLUR", True)
    polygon_padding_ratio: float = float(os.getenv("POLYGON_PADDING_RATIO", "0.25"))
    snapshot_dir: str = os.getenv("SNAPSHOT_DIR", "snapshots")
    weights_dir: str = os.getenv("WEIGHTS_DIR", "weights")
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.4"))


settings = Settings()
