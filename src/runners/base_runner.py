from pathlib import Path
from typing import Any, Dict, Optional

from utils.logging import setup_logger
from utils.metrics import write_metrics_json
from pipelines.train_eval import run_train_eval
import time
import json
import yaml
from utils.env_info import compute_env_hash


class BaseRunner:
    """Skeleton runner handling hooks, logging, and checkpoint stubs."""

    def __init__(self, work_dir: Path, config: Dict[str, Any], config_path: Optional[Path] = None, env_hash: str = ""):
        self.work_dir = work_dir
        self.config = config
        # Use the provided work_dir directly (e.g., work_dirs/face_swap/<name>/)
        self.task_dir = self.work_dir
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.__class__.__name__)
        self.checkpoints_dir = self.task_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_config(config_path=config_path)

    def _snapshot_config(self, config_path: Optional[Path]) -> None:
        self.task_dir.mkdir(parents=True, exist_ok=True)
        # Only write YAML and PY versions of config
        (self.task_dir / "config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False))
        (self.task_dir / "config.py").write_text(f"config = {json.dumps(self.config, indent=2, ensure_ascii=True)}\n")

    def before_run(self) -> None:
        self.logger.info('{"event": "before_run"}')

    def after_run(self) -> None:
        self.logger.info('{"event": "after_run"}')

    def save_checkpoint(self, epoch: int = 1, name: Optional[str] = None) -> Path:
        ckpt_name = name or f"epoch_{epoch:02d}.pt"
        path = self.checkpoints_dir / ckpt_name
        path.write_text("checkpoint-placeholder")
        self.logger.info(f'{{"event": "checkpoint_saved", "path": "{path}"}}')
        return path

    def train(self) -> Dict[str, Any]:
        self.before_run()
        self.logger.info('{"event": "train_start"}')
        metrics = run_train_eval(
            self.config,
            self.task_dir,
            mode="train",
            checkpoint_dir=self.checkpoints_dir,
        )
        write_metrics_json(self.task_dir, metrics, filename="metrics.train.json")
        self.after_run()
        return metrics

    def evaluate(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        self.before_run()
        self.logger.info('{"event": "eval_start"}')
        metrics = run_train_eval(
            self.config,
            self.task_dir,
            mode="eval",
        )
        self.after_run()
        return metrics


def build_runner(work_dir: Path, config: Dict[str, Any], config_path: Optional[Path] = None, env_hash: str = "") -> BaseRunner:
    return BaseRunner(work_dir=work_dir, config=config, config_path=config_path)
