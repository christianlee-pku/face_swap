import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DatasetManifest:
    version: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    splits: Dict[str, List[str]] = field(default_factory=dict)
    checksums: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    pairs: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        data = json.loads(path.read_text())
        return cls(**data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=True))

    def bump_version(self, new_version: str) -> None:
        self.version = new_version
