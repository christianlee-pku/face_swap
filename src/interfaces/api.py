from pathlib import Path
from typing import Any, Dict, Optional

from runners.base_runner import build_runner
from utils.config import prepare_run


def train(config_path: Path, work_dir: Optional[Path] = None) -> None:
    ctx = prepare_run(config_path, work_dir)
    runner = build_runner(ctx["work_dir"], ctx["config"])
    runner.train()


def evaluate(config_path: Path, work_dir: Optional[Path] = None) -> Dict[str, Any]:
    ctx = prepare_run(config_path, work_dir)
    runner = build_runner(ctx["work_dir"], ctx["config"])
    return runner.evaluate()


def infer(config_path: Path, sources: Any, targets: Any, work_dir: Optional[Path] = None) -> Dict[str, Any]:
    # Placeholder that echoes inputs
    _ = prepare_run(config_path, work_dir)
    return {"sources": sources, "targets": targets, "status": "not_implemented"}
