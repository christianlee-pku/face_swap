from pathlib import Path

from src.pipelines.eval_only import run_eval_only


def test_eval_only_pipeline(tmp_path: Path):
    cfg = {
        "name": "eval-only-test",
        "dataset": {"type": "LFWDataset", "root": str(tmp_path), "manifest": str(tmp_path / "manifest.json"), "split": "val"},
        "model": {"type": "UNetFaceSwap", "channels": 8},
        "loss": {"type": "FaceSwapLoss"},
    }
    (tmp_path / "manifest.json").write_text('{"version":"1.0","items":[],"splits":{"val":[]}}')
    metrics = run_eval_only(cfg, tmp_path / "wd")
    assert "identity_accuracy" in metrics
    assert (tmp_path / "wd" / "metrics.eval_only.json").exists()
