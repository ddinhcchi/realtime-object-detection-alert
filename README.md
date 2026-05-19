# 🛡️ Realtime Object Detection & Smart Alert

End-to-end Computer Vision demo: webcam → YOLOv8 inference on Apple Silicon (MPS) → bounding box overlay → Telegram alert with snapshot when a watched class is detected. Configurable per-class cooldown to avoid alert spam.

![demo](demo/demo.gif)

> Replace `demo/demo.gif` with a 5–15s screen recording: start the app, pick `person` as watch class, walk in front of the webcam, see the red box + Telegram message arrive.

---

## Why this project

Most CV demos stop at "model predicts the class". Production CV is the **rest**: keeping latency under a target, deciding *when* to alert, throttling noisy detections, and packaging the system so an end-user can run it. This repo shows that full path.

---

## Features

- **YOLOv8 nano** (COCO 80 classes) with device auto-select: `mps` on Apple Silicon, `cuda` on NVIDIA, `cpu` fallback
- **Live webcam UI** via Streamlit + `streamlit-webrtc`
- **Watch list** — alert only on classes you care about
- **Cooldown per class** (default 10s) prevents flooding
- **Telegram delivery** with bbox-rendered snapshot
- **Honest benchmark script** (no fabricated GPU numbers)
- **Dockerfile** for CPU-only deploys (Streamlit Cloud, Fly.io, etc.)

---

## Benchmark — Apple M4 (10-core, 16GB)

YOLOv8n, 640×640 synthetic frame, 50 runs after 5-run warmup. Measured by [`scripts/benchmark.py`](scripts/benchmark.py).

| Device | p50 latency | p95 latency | mean | throughput |
|---|---:|---:|---:|---:|
| **MPS (Apple GPU)** | **10.5 ms** | 21.1 ms | 11.7 ms | **~95 FPS** |
| CPU (fallback) | 31.4 ms | 40.0 ms | 32.6 ms | ~32 FPS |

Reproduce on your machine:

```bash
PYTHONPATH=. python scripts/benchmark.py
```

---

## Quick start (local)

Requires Python 3.11+ (tested on 3.14). Webcam permission for your terminal/IDE.

```bash
git clone <this-repo>
cd realtime-object-detection-alert
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit with your Telegram bot token + chat id
streamlit run app.py
```

Open http://localhost:8501 → click **Start** → pick classes to watch → toggle alert.

### Getting Telegram credentials (5 min)

1. Open Telegram, chat with **@BotFather**, send `/newbot`, follow prompts → get bot token
2. Start a chat with your new bot, send any message
3. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates` → copy `chat.id` from the JSON
4. Paste both into `.env`

---

## Run with Docker (CPU only)

The Docker image runs CPU inference (Docker Desktop on macOS cannot pass through Apple GPU). Useful for cloud deploys.

```bash
docker build -t obj-detect-alert .
docker run -p 8501:8501 --env-file .env obj-detect-alert
```

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│ Webcam frame │ -> │ YOLOv8 (MPS) │ -> │ Watch-list  │ -> │  Telegram  │
└──────────────┘    └──────────────┘    │  + cooldown │    │  + snap on │
                                        └─────────────┘    │    disk    │
                                                            └────────────┘
```

Code layout:

| File | Responsibility |
|---|---|
| [`src/detector.py`](src/detector.py) | YOLO wrapper, device resolution, latency tracking |
| [`src/alert.py`](src/alert.py) | Telegram client, per-class cooldown, snapshot save |
| [`src/draw.py`](src/draw.py) | Bbox + HUD overlay |
| [`src/config.py`](src/config.py) | `.env`-driven settings |
| [`app.py`](app.py) | Streamlit + WebRTC entrypoint |
| [`scripts/benchmark.py`](scripts/benchmark.py) | MPS vs CPU benchmark |

---

## Sample watch lists

| Use case | Classes |
|---|---|
| Worksite safety | `person`, `bicycle`, `motorcycle`, `car`, `truck` |
| Home / pets | `dog`, `cat`, `bird` |
| Office | `laptop`, `cell phone`, `book`, `cup` |

All 80 COCO classes are available — see `detector.class_names` at runtime.

---

## Roadmap / ideas

- Custom dataset fine-tuning recipe (YOLO format)
- Zone-of-interest polygon (alert only inside a drawn region)
- ONNX / CoreML export script for true edge deployment
- WebSocket sink instead of Telegram for system integration

---

## License

MIT
