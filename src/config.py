import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    alert_cooldown_seconds: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "10"))
    device: str = os.getenv("DEVICE", "mps")
    model_weights: str = os.getenv("MODEL_WEIGHTS", "yolov8n.pt")
    snapshot_dir: str = os.getenv("SNAPSHOT_DIR", "snapshots")
    default_watch_classes: list[str] = field(default_factory=lambda: ["person"])
    default_conf_threshold: float = 0.5


settings = Settings()
