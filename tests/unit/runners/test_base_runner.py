from pathlib import Path

from src.runners.base_runner import BaseRunner


def test_runner_creates_checkpoint_dir(tmp_path: Path):
    runner = BaseRunner(work_dir=tmp_path, config={"name": "test"})
    assert runner.checkpoints_dir.exists()


def test_runner_train_saves_checkpoint(tmp_path: Path):
    runner = BaseRunner(work_dir=tmp_path, config={"name": "test"})
    runner.train()
    assert (tmp_path / "checkpoints" / "last.pth").exists()


def test_runner_evaluate_returns_metrics(tmp_path: Path):
    runner = BaseRunner(work_dir=tmp_path, config={"name": "test"})
    metrics = runner.evaluate()
    assert "identity_accuracy" in metrics
