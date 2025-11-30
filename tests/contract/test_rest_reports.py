import importlib
import pytest


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_reports_endpoint_exists():
    rest = importlib.import_module("src.interfaces.rest")
    assert hasattr(rest, "app")
    # FastAPI app should include reports route
    routes = [r.path for r in rest.app.routes]  # type: ignore[attr-defined]
    assert "/reports/{run_id}" in routes
