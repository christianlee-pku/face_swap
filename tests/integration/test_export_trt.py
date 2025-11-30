import importlib
from pathlib import Path

import pytest


@pytest.mark.skip(reason="TensorRT not available in test environment")
def test_export_to_tensorrt(tmp_path: Path):
    trt = importlib.import_module("src.exporters.tensorrt_exporter")
    engine_path = trt.export_to_tensorrt(tmp_path / "model.onnx", tmp_path / "export")
    assert engine_path.exists()
