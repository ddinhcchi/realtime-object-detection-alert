import time

import cv2
import streamlit as st

from src.alert import TelegramAlerter
from src.blur import FaceBlurrer
from src.config import settings
from src.detector import Detector
from src.draw import draw_hud, draw_tracks, draw_zone
from src.rtsp import RtspReader
from src.zone import DwellTracker, centered_polygon

st.set_page_config(page_title="RTSP Zone Intrusion Alerts", layout="wide")
st.title("🛡️ RTSP Zone-Intrusion Detector")
st.caption(
    "YOLOv8 + ByteTrack on Apple Silicon. A polygon zone in the centre of the "
    "frame fires a Telegram alert when a watched class lingers for ≥ "
    f"{int(settings.dwell_threshold_seconds)}s."
)


@st.cache_resource
def get_detector() -> Detector:
    return Detector(weights=settings.model_weights, device=settings.device)


@st.cache_resource
def get_blurrer() -> FaceBlurrer:
    return FaceBlurrer(weights_dir=settings.weights_dir)


@st.cache_resource
def get_reader(url: str) -> RtspReader:
    reader = RtspReader(url)
    reader.start()
    return reader


detector = get_detector()
watch_cid = detector.class_name_to_id(settings.watch_class)
classes_filter = [watch_cid] if watch_cid is not None else None

with st.sidebar:
    st.subheader("Source")
    rtsp_url = st.text_input(
        "RTSP URL",
        value=settings.rtsp_url,
        type="password",
        help="rtsp://user:pass@host:554/path",
    )
    st.subheader("Detection")
    conf = st.slider("Confidence", 0.1, 0.95, settings.conf_threshold, 0.05)
    watch_class = st.selectbox(
        "Watch class",
        options=sorted(detector.class_names.values()),
        index=sorted(detector.class_names.values()).index(settings.watch_class),
    )
    dwell_threshold = st.slider(
        "Dwell threshold (s)", 1.0, 60.0, settings.dwell_threshold_seconds, 1.0
    )
    st.subheader("Privacy")
    enable_blur = st.toggle("Blur faces (YuNet)", value=settings.enable_face_blur)
    st.subheader("Zone")
    pad_ratio = st.slider(
        "Polygon inset", 0.05, 0.45, settings.polygon_padding_ratio, 0.05
    )
    st.subheader("Alert")
    enable_alert = st.toggle("Telegram alert on dwell", value=True)
    st.caption(
        "Configured ✅" if settings.telegram_bot_token and settings.telegram_chat_id
        else "Set TELEGRAM_BOT_TOKEN + CHAT_ID in .env"
    )

if not rtsp_url:
    st.warning("Enter an RTSP URL in the sidebar to start.")
    st.stop()

if "running" not in st.session_state:
    st.session_state.running = False

c1, c2 = st.columns([1, 1])
if c1.button("▶ Start", use_container_width=True, type="primary"):
    st.session_state.running = True
if c2.button("⏹ Stop", use_container_width=True):
    st.session_state.running = False

frame_slot = st.empty()
status_slot = st.empty()

if st.session_state.running:
    reader = get_reader(rtsp_url)
    blurrer = get_blurrer() if enable_blur else None
    alerter = TelegramAlerter(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        snapshot_dir=settings.snapshot_dir,
    )
    dwell = DwellTracker(
        threshold_seconds=dwell_threshold,
        cooldown_seconds=settings.alert_cooldown_seconds,
    )

    polygon = None
    last_render = 0.0
    while st.session_state.running:
        frame = reader.read()
        if frame is None:
            status_slot.warning("Waiting for first frame…")
            time.sleep(0.2)
            continue

        if polygon is None:
            h, w = frame.shape[:2]
            polygon = centered_polygon(w, h, pad_ratio)

        tracks = detector.track(frame, conf=conf, classes=classes_filter)
        events = dwell.update(tracks, polygon, watch_class)

        rendered = frame.copy()
        if blurrer is not None:
            rendered = blurrer.apply(rendered)
        hit = any(
            dwell.dwell_for(t.track_id) > 0
            for t in tracks
            if t.track_id is not None and t.class_name == watch_class
        )
        draw_zone(rendered, polygon, hit=hit)
        draw_tracks(rendered, tracks, dwell.dwell_for)
        draw_hud(rendered, detector.device, detector.last_latency_ms, len(tracks))

        if enable_alert and events and alerter.is_configured:
            for ev in events:
                alerter.send_dwell(
                    track_id=ev.track_id,
                    class_name=ev.class_name,
                    dwell_seconds=ev.dwell_seconds,
                    frame_bgr=rendered,
                )
                status_slot.success(
                    f"Alert sent: track #{ev.track_id} dwelt {ev.dwell_seconds:.1f}s"
                )

        # cap UI to ~20 FPS to keep Streamlit responsive
        now = time.time()
        if now - last_render >= 0.05:
            frame_slot.image(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
            last_render = now
else:
    frame_slot.info("Press **Start** to begin streaming.")
