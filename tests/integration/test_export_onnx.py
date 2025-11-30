from pathlib import Path

from src.exporters.onnx_exporter import export_to_onnx, load_model_from_config


def test_export_to_onnx(tmp_path: Path):
    model = load_model_from_config({"type": "UNetFaceSwap"})
    checkpoint = tmp_path / "ckpt.pth"
    checkpoint.write_text("ckpt")
    onnx_path = export_to_onnx(model, checkpoint, tmp_path / "export")
    assert onnx_path.exists()
