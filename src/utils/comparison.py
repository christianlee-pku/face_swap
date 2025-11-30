from typing import Dict


def compare_runs(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    """Compute metric deltas candidate - baseline."""
    keys = set(baseline.keys()) | set(candidate.keys())
    return {k: candidate.get(k, 0.0) - baseline.get(k, 0.0) for k in keys}
