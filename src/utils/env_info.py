import hashlib
from pathlib import Path
from typing import Iterable


DEFAULT_ENV_FILES = ["environment.lock.yml", "environment.yml"]


def _hash_files(files: Iterable[Path]) -> str:
    hasher = hashlib.sha256()
    for path in files:
        if path.exists():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def compute_env_hash(env_files: Iterable[str] = DEFAULT_ENV_FILES) -> str:
    paths = [Path(f) for f in env_files]
    return _hash_files(paths)
