from pathlib import Path

from src.utils.config import load_config, prepare_run
from src.utils.env_info import compute_env_hash
from src.utils.workdir import ensure_work_dir


def test_load_config(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("a: 1\n")
    loaded = load_config(cfg_path)
    assert loaded.data["a"] == 1
    assert loaded.path == cfg_path


def test_ensure_work_dir_with_name(tmp_path: Path):
    cfg = {"name": "test-exp"}
    wd = ensure_work_dir(None, cfg)
    assert wd.exists()
    assert "test-exp" in wd.name


def test_prepare_run_writes_snapshot(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: test-exp\nseed: 1\n")
    ctx = prepare_run(cfg_path)
    assert (ctx["work_dir"] / "config.snapshot.json").exists()
    assert (ctx["work_dir"] / "env.hash").exists()


def test_compute_env_hash_handles_missing_files(tmp_path: Path):
    h = compute_env_hash(["missing1.yml", "missing2.yml"])
    assert isinstance(h, str)
    assert len(h) == 64
