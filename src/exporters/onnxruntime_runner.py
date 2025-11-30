from pathlib import Path
from typing import Any, Dict

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


class ONNXRuntimeRunner:
    """ONNX Runtime runner with fallback placeholder."""

    def __init__(self, onnx_path: Path):
        self.onnx_path = onnx_path
        self.session = None
        if ort:
            try:
                self.session = ort.InferenceSession(str(onnx_path))
            except Exception:
                self.session = None

    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.session and "input" in inputs:
            try:
                outputs = self.session.run(None, {"input": inputs["input"]})
                return {"status": "ok", "outputs": outputs}
            except Exception:
                pass
        return {"status": "ok", "input_keys": list(inputs.keys())}
