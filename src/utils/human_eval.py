from pathlib import Path
from typing import Dict, List
import json


def record_human_ratings(work_dir: Path, samples: List[Dict[str, str]], filename: str = "human_eval.json") -> Path:
    """Store placeholder MOS-style ratings metadata for later manual collection."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / filename
    path.write_text(json.dumps(samples, indent=2, ensure_ascii=True))
    return path
