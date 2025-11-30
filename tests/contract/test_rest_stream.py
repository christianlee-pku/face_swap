import importlib
import pytest


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_rest_stream_route_exists():
    rest = importlib.import_module("src.interfaces.rest")
    routes = [r.path for r in rest.app.routes]  # type: ignore[attr-defined]
    assert "/face-swap/stream" in routes
