import time
from typing import Tuple


def measure_latency_fps() -> Tuple[float, float]:
    """Return placeholder latency (ms) and FPS measurements."""
    start = time.time()
    time.sleep(0.001)
    latency_ms = (time.time() - start) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    return round(latency_ms, 3), round(fps, 3)
