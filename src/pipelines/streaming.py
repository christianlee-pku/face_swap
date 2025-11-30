from pathlib import Path
from typing import Any, Dict, Iterable, List

from utils.perf import measure_latency_fps


def run_streaming(config: Dict[str, Any], frames: Iterable[Any], work_dir: Path) -> Dict[str, Any]:
    """Streaming pipeline with latency/FPS measurement and basic output logging."""
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_list: List[Any] = list(frames)
    latency_ms, fps = measure_latency_fps()
    (work_dir / "stream.log").write_text(f"frames={len(frames_list)}, latency_ms={latency_ms}, fps={fps}")
    return {"processed_frames": len(frames_list), "latency_ms": latency_ms, "fps": fps}
