"""Measure inference latency on a synthetic 640x640 frame, MPS vs CPU."""
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import Detector  # noqa: E402


def bench(device: str, runs: int = 50) -> dict:
    det = Detector(device=device)
    # warmup
    frame = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    for _ in range(5):
        det.predict(frame)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        det.predict(frame)
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "device": det.device,
        "runs": runs,
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(0.95 * runs) - 1], 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "fps": round(1000 / statistics.median(latencies), 1),
    }


if __name__ == "__main__":
    for dev in ("mps", "cpu"):
        try:
            print(bench(dev))
        except Exception as e:
            print({"device": dev, "error": str(e)})
