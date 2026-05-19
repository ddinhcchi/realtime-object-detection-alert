# 🛡️ RTSP Zone-Intrusion Detector

Live IP-camera feed → YOLOv8 detection → ByteTrack multi-object tracking → polygon zone dwell timer → Telegram alert with snapshot. Built for real CCTV / Hikvision / Dahua deployments, not webcam demos.

![demo](demo/demo.gif)

> Demo GIF (faces blurred with YuNet for privacy): a person enters the orange zone, the dwell counter ticks, after 10 s the zone turns red and a snapshot lands in Telegram.

---

## Why this project

The job an SME actually wants done is *"alert me when someone lingers near the back door for more than 10 seconds"* — not *"classify this image"*. That requires four things glued together:

1. A reliable **RTSP reader** that recovers from network drops.
2. **Detection** under varied lighting at >15 FPS on commodity hardware.
3. **Multi-object tracking** so the dwell timer follows a person across frames.
4. **Stateful alert logic** with per-track cooldowns so you don't spam ops at 3 a.m.

This repo demonstrates all four in a single Streamlit app.

---

## Features

- **RTSP input** with a threaded reader that drops stale frames and auto-reconnects
- **YOLOv8 nano** detection with device auto-select: `mps` on Apple Silicon → `cuda` → `cpu`
- **ByteTrack** multi-object tracking via Ultralytics' built-in tracker
- **Polygon zone** centred in the frame (configurable inset)
- **Per-track dwell timer**: alert fires when a tracked ID stays in zone for ≥ N seconds
- **Per-track cooldown**: same person re-entering won't spam — controlled by `ALERT_COOLDOWN_SECONDS`
- **Telegram delivery** with bbox-rendered snapshot + caption (class, track id, dwell time)
- **Face blur (YuNet)** toggle — open-source ONNX face detector, weights auto-download (~230 KB), for privacy in demo recordings
- **Honest benchmark** on Apple M4 included below

---

## Benchmark — Apple M4 (10-core, 16 GB)

YOLOv8n, 640×640 synthetic frame, 50 runs after 5-run warm-up. Reproduce with `PYTHONPATH=. python scripts/benchmark.py`.

| Device | p50 | p95 | mean | throughput |
|---|---:|---:|---:|---:|
| **MPS (Apple GPU)** | **10.5 ms** | 21.1 ms | 11.7 ms | **~95 FPS** |
| CPU fallback | 31.4 ms | 40.0 ms | 32.6 ms | ~32 FPS |

End-to-end with ByteTrack + zone test + YuNet blur the pipeline still clears 25–30 FPS on M4 — well above any IP camera's 15-FPS output.

---

## Architecture

```
┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌──────────────┐
│ RTSP stream  │ → │ Threaded reader  │ → │ YOLOv8 +     │ → │ DwellTracker │
│ (IP camera)  │   │ (latest-frame)   │   │ ByteTrack    │   │ per track_id │
└──────────────┘   └──────────────────┘   └──────────────┘   └──────┬───────┘
                                                                    │
                              ┌─────────────────────────────────────┘
                              ▼
                ┌────────────────────────────┐
                │ Polygon-in-zone test       │  if dwell ≥ N s and not on cooldown
                │ for each tracked person    │  → Telegram snapshot + caption
                └────────────────────────────┘
```

### Code layout

| File | Responsibility |
|---|---|
| [`src/rtsp.py`](src/rtsp.py) | Threaded RTSP reader, TCP transport, auto-reconnect |
| [`src/detector.py`](src/detector.py) | YOLOv8 + ByteTrack tracker (`model.track(persist=True)`) |
| [`src/zone.py`](src/zone.py) | Centred polygon + `DwellTracker` (per-id timer + cooldown) |
| [`src/blur.py`](src/blur.py) | YuNet face blur (privacy for demos) |
| [`src/alert.py`](src/alert.py) | Telegram client, snapshot save, dwell-aware caption |
| [`src/draw.py`](src/draw.py) | Zone overlay, track boxes, HUD |
| [`src/config.py`](src/config.py) | `.env`-driven settings |
| [`app.py`](app.py) | Streamlit UI |

---

## Quick start

Requires Python 3.11+. Tested on Apple M4 / Python 3.14.

```bash
git clone <this-repo>
cd realtime-object-detection-alert
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env:
#   RTSP_URL=rtsp://user:pass@CAMERA_IP:554/stream
#   TELEGRAM_BOT_TOKEN=...
#   TELEGRAM_CHAT_ID=...

streamlit run app.py
```

Open http://localhost:8501 → **Start** → people walking into the centre polygon trigger an alert after 10 s.

### Configuration knobs (sidebar + `.env`)

| Setting | Default | Effect |
|---|---|---|
| `RTSP_URL` | — | Camera stream URL |
| `DWELL_THRESHOLD_SECONDS` | 10 | How long a person must stay in the zone before alerting |
| `ALERT_COOLDOWN_SECONDS` | 30 | Per-track cooldown after firing |
| `WATCH_CLASS` | `person` | Any COCO class — `car`, `truck`, `dog`, etc. |
| `POLYGON_PADDING_RATIO` | 0.25 | Fraction of frame inset on each side (smaller = bigger zone) |
| `ENABLE_FACE_BLUR` | `true` | Run YuNet over the rendered frame |

### Getting Telegram credentials (5 min)

1. Open Telegram, message **@BotFather**, `/newbot` → get bot token
2. Start a chat with your new bot, send any message
3. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates` → copy `chat.id`
4. Paste both into `.env`

---

## Run with Docker (CPU only)

Docker Desktop on macOS cannot pass through Apple GPU, so the image runs CPU inference — fine for any cloud host.

```bash
docker build -t rtsp-zone-alert .
docker run --rm -p 8501:8501 --env-file .env rtsp-zone-alert
```

---

## Why ByteTrack (not just IoU matching)

A naïve "match by IoU each frame" tracker loses identity whenever the detector flickers — and YOLO does flicker, especially at lower confidence thresholds. ByteTrack keeps low-confidence detections as candidate matches for existing tracks, so a person briefly occluded behind a chair doesn't get a new ID when they reappear. For dwell-time logic that matters: a flipped ID resets the timer.

## Why YuNet for face blur

For demo recordings you want detections that fire even on profile / partially occluded faces, not just the strict frontal-face detector that Haar cascade gives you. [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) is a 230 KB ONNX model that catches both. The weights auto-download on first run.

---

## Roadmap

- Draw / drag custom polygon in the Streamlit UI (instead of axis-aligned centred rect)
- Persist alert history to SQLite + dashboard tab
- Multi-camera support (one process per RTSP)
- ONNX / CoreML export of YOLO for true edge deployment

---

## License

MIT
