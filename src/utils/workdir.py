import time
from pathlib import Path
from typing import Any, Dict, Optional


def _make_name(config: Dict[str, Any]) -> str:
    base = config.get("name") or config.get("experiment_name")
    if base:
        return str(base)
    ts = int(time.time())
    return f"exp-{ts}"


def ensure_work_dir(
    work_dir: Optional[Path], config: Dict[str, Any], config_path: Optional[Path] = None
) -> Path:
    target = work_dir
    if target is None:
        base_name = _make_name(config)
        target = Path("work_dirs") / "face_swap" / base_name
    target.mkdir(parents=True, exist_ok=True)
    # No files at root; task-specific files live under task subdirs.
    return target
