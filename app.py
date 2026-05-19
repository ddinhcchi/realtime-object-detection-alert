import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from src.alert import TelegramAlerter
from src.config import settings
from src.detector import Detector
from src.draw import draw_detections

st.set_page_config(page_title="Realtime Object Detection & Alert", layout="wide")
st.title("🛡️ Real-Time Object Detection & Smart Alert")
st.caption(
    "YOLOv8 on Apple Silicon (MPS). Detect objects from your webcam, "
    "trigger Telegram alerts with cooldown."
)


@st.cache_resource
def get_detector() -> Detector:
    return Detector(weights=settings.model_weights, device=settings.device)


detector = get_detector()
all_classes = sorted(detector.class_names.values())

with st.sidebar:
    st.subheader("Detection settings")
    watch = st.multiselect(
        "Watch list (alert on these classes)",
        options=all_classes,
        default=[c for c in settings.default_watch_classes if c in all_classes],
    )
    conf = st.slider("Confidence threshold", 0.1, 0.95, settings.default_conf_threshold, 0.05)
    enable_alert = st.toggle("Send Telegram alert on detection", value=False)
    st.markdown("---")
    st.caption(f"Device: **{detector.device.upper()}**")
    st.caption(f"Cooldown: {settings.alert_cooldown_seconds}s per class")

alerter = TelegramAlerter(
    bot_token=settings.telegram_bot_token,
    chat_id=settings.telegram_chat_id,
    cooldown_seconds=settings.alert_cooldown_seconds,
    snapshot_dir=settings.snapshot_dir,
)

if enable_alert and not alerter.is_configured:
    st.sidebar.warning("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable alerts.")


class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.detector = detector
        self.alerter = alerter
        self.conf = settings.default_conf_threshold
        self.watch: set[str] = set()
        self.enable_alert = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        detections = self.detector.predict(img, conf=self.conf)
        rendered = draw_detections(
            img,
            detections,
            self.watch,
            self.detector.last_latency_ms,
            self.detector.device,
        )
        if self.enable_alert and self.watch:
            for det in detections:
                if det.class_name in self.watch:
                    self.alerter.send(det.class_name, det.confidence, img)
                    break
        return av.VideoFrame.from_ndarray(rendered, format="bgr24")


ctx = webrtc_streamer(
    key="detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.conf = conf
    ctx.video_processor.watch = set(watch)
    ctx.video_processor.enable_alert = enable_alert and alerter.is_configured

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("How it works")
    st.markdown(
        "- YOLOv8 nano (COCO 80 classes) loaded once on first run\n"
        "- Each webcam frame → inference on **MPS** → bbox overlay\n"
        "- If a watched class is detected and alerts are on, fire Telegram "
        "with snapshot (cooldown per class)\n"
    )
with col2:
    st.subheader("Sample watch lists")
    st.markdown(
        "- **Safety**: person, bicycle, motorcycle, car, truck\n"
        "- **Pets**: dog, cat, bird\n"
        "- **Office**: laptop, cell phone, book, cup\n"
    )
