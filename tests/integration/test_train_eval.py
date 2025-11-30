from pathlib import Path

from src.runners.base_runner import BaseRunner
from src.utils.config import prepare_run


def test_train_eval_pipeline(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: train-eval-test",
                "dataset:",
                "  type: LFWDataset",
                f"  root: {tmp_path}",
                f"  manifest: {tmp_path/'manifest.json'}",
                "  split: train",
                "model:",
                "  type: UNetFaceSwap",
                "loss:",
                "  type: FaceSwapLoss",
            ]
        )
    )
    # empty manifest to allow dataset load
    (tmp_path / "manifest.json").write_text('{"version":"1.0","items":[],"splits":{"train":[]}}')
    ctx = prepare_run(cfg_path)
    runner = BaseRunner(ctx["work_dir"], ctx["config"])
    metrics = runner.train()
    assert "identity_accuracy" in metrics
    eval_metrics = runner.evaluate()
    assert "psnr" in eval_metrics
