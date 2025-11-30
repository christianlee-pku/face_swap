import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .env_info import compute_env_hash
from .workdir import ensure_work_dir


@dataclass
class LoadedConfig:
    data: Dict[str, Any]
    path: Path


def load_config(path: Path) -> LoadedConfig:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return LoadedConfig(data=cfg, path=path)


def apply_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def snapshot_config(cfg: Dict[str, Any], work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = work_dir / "config.snapshot.json"
    snapshot_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=True))
    return snapshot_path


def prepare_run(config_path: Path, work_dir: Optional[Path] = None) -> Dict[str, Any]:
    loaded = load_config(config_path)
    seed = loaded.data.get("seed")
    apply_seed(seed)

    target_work_dir = ensure_work_dir(work_dir, loaded.data, config_path)
    # Keep root clean; task runner will snapshot config under task dir.
    env_hash = compute_env_hash()
    return {"config": loaded.data, "work_dir": target_work_dir, "env_hash": env_hash, "config_path": config_path}
