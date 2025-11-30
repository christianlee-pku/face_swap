from pathlib import Path
from typing import Dict, Iterable, List

from .metrics import write_metrics_json, write_metrics_csv


def save_metric_graphs(work_dir: Path, metrics: Dict[str, float], filename_prefix: str = "metrics") -> None:
    """Persist metrics (placeholder for plotting)."""
    write_metrics_json(work_dir, metrics, filename=f"{filename_prefix}.json")
    write_metrics_csv(work_dir, [metrics], filename=f"{filename_prefix}.csv")


def save_comparisons(work_dir: Path, rows: Iterable[Dict[str, float]], filename_prefix: str = "comparisons") -> None:
    write_metrics_csv(work_dir, list(rows), filename=f"{filename_prefix}.csv")


def save_sample_gallery(work_dir: Path, samples: List[Dict[str, str]], filename: str = "samples.json") -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / filename
    write_metrics_json(work_dir, {"samples": samples}, filename=filename)
    return path
