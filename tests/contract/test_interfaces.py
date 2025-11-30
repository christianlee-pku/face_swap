import importlib
import sys
from pathlib import Path

import pytest


def test_cli_parses_train(monkeypatch, capsys, tmp_path: Path):
    cli = importlib.import_module("src.interfaces.cli")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("name: cli-test\n")
    argv = ["train", "--config", str(cfg), "--work-dir", str(tmp_path / "wd")]
    monkeypatch.setattr(sys, "argv", ["prog"] + argv)
    cli.main()
    out = (tmp_path / "wd" / "checkpoints" / "last.pth")
    assert out.exists()


def test_api_train_and_eval(tmp_path: Path):
    api = importlib.import_module("src.interfaces.api")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("name: api-test\n")
    api.train(cfg, tmp_path / "wd-train")
    metrics = api.evaluate(cfg, tmp_path / "wd-eval")
    assert "identity_accuracy" in metrics


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_rest_app_imports():
    rest = importlib.import_module("src.interfaces.rest")
    assert rest.app is not None
