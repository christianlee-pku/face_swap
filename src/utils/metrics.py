import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def write_metrics_json(output_dir: Path, metrics: Mapping[str, Any], filename: str = "metrics.json") -> Path:
    """Persist metrics dict to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2))
    return path


def write_metrics_csv(output_dir: Path, rows: Iterable[Mapping[str, Any]], filename: str = "metrics.csv") -> Path:
    """Persist metrics rows to CSV with header inferred from first row."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    rows = list(rows)
    if not rows:
        path.write_text("")
        return path
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        line = ",".join(str(row.get(h, "")) for h in headers)
        lines.append(line)
    path.write_text("\n".join(lines))
    return path
