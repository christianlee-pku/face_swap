from pathlib import Path
from typing import Any, Dict

from .onnx_exporter import load_model_from_config, export_to_onnx
from .tensorrt_exporter import export_to_tensorrt
from ..utils.perf import measure_latency_fps


def benchmark_edge(config: Dict[str, Any], checkpoint: Path, export_dir: Path, target: str = "jetson") -> Dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_from_config(config)
    onnx_path = export_to_onnx(model, checkpoint, export_dir)
    trt_path = export_to_tensorrt(onnx_path, export_dir, precision="fp16")
    latency_ms, fps = measure_latency_fps()
    results = {"target": target, "fps": fps, "latency_ms": latency_ms, "onnx": str(onnx_path), "tensorrt": str(trt_path)}
    (export_dir / "benchmark.json").write_text(str(results))
    return results
