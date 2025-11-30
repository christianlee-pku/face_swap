from pathlib import Path
from typing import Any, Dict

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..registry import MODELS


def load_model_from_config(config: Dict[str, Any]) -> Any:
    """Instantiate model from registry when available."""
    model_cfg = config if isinstance(config, dict) else {"type": str(config)}
    if MODELS and "type" in model_cfg:
        try:
            return MODELS.build(model_cfg)
        except Exception:
            return model_cfg
    return model_cfg


def export_to_onnx(model: Any, checkpoint: Path, export_dir: Path, dynamic_axes: bool = True) -> Path:
    """Export model to ONNX if torch available; otherwise write placeholder."""
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "model.onnx"
    if torch and hasattr(model, "state_dict"):
        dummy = torch.randn(1, 3, 112, 112)
        try:
            torch.onnx.export(
                model,
                dummy,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_axes else None,
                opset_version=12,
            )
            return onnx_path
        except Exception:
            pass
    onnx_path.write_text("onnx-placeholder")
    return onnx_path
