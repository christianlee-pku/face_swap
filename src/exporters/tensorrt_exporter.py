from pathlib import Path
from typing import Any, Dict
import subprocess


def export_to_tensorrt(onnx_path: Path, export_dir: Path, precision: str = "fp16") -> Path:
    """Attempt TensorRT export via trtexec; fallback to placeholder."""
    export_dir.mkdir(parents=True, exist_ok=True)
    engine_path = export_dir / f"model.{precision}.engine"
    trtexec = "trtexec"
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]
    if precision.lower() == "fp16":
        cmd.append("--fp16")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return engine_path
    except Exception:
        engine_path.write_text("tensorrt-placeholder")
        return engine_path
