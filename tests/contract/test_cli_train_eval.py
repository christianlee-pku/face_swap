import sys
from pathlib import Path

from src.interfaces import cli


def test_cli_train_and_eval(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"version":"1.0","items":[],"splits":{"train":[]}}')
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-train-eval",
                "dataset:",
                "  type: LFWDataset",
                f"  root: {tmp_path}",
                f"  manifest: {manifest}",
                "  split: train",
                "model:",
                "  type: UNetFaceSwap",
                "loss:",
                "  type: FaceSwapLoss",
            ]
        )
    )

    # train
    argv_train = ["train", "--config", str(cfg), "--work-dir", str(tmp_path / "wd-train")]
    monkeypatch.setattr(sys, "argv", ["prog"] + argv_train)
    cli.main()
    assert (tmp_path / "wd-train" / "checkpoints" / "last.pth").exists()

    # eval
    argv_eval = ["eval", "--config", str(cfg), "--work-dir", str(tmp_path / "wd-eval")]
    monkeypatch.setattr(sys, "argv", ["prog"] + argv_eval)
    cli.main()
    assert (tmp_path / "wd-eval" / "metrics.eval.json").exists()
