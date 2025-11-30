from pathlib import Path
from typing import Dict, Any

from .manifest import DatasetManifest


def update_dataset(manifest_path: Path, new_version: str, changes: Dict[str, Any]) -> Path:
    """Version bump and changelog entry for dataset updates."""
    manifest = DatasetManifest.load(manifest_path) if manifest_path.exists() else DatasetManifest(version="0.0.0")
    manifest.bump_version(new_version)
    manifest.meta.setdefault("changelog", []).append(changes)
    manifest.save(manifest_path)
    return manifest_path
