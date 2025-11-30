from typing import Any, Dict

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover - fastapi may be optional
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore

from interfaces.api import evaluate, infer, train
from pipelines.streaming import run_streaming


if FastAPI:
    app = FastAPI(title="Face Swap API", version="0.1.0")

    class SwapRequest(BaseModel):
        sources: Any
        targets: Any
        config: str
        work_dir: str | None = None

    class EvalRequest(BaseModel):
        config: str
        work_dir: str | None = None

    class StreamRequest(BaseModel):
        config: str
        frames: list
        work_dir: str | None = None

    @app.post("/face-swap/batch")
    def swap(req: SwapRequest) -> Dict[str, Any]:  # type: ignore[override]
        return infer(req.config, req.sources, req.targets, work_dir=req.work_dir)

    @app.post("/face-swap/train")
    def train_endpoint(req: EvalRequest) -> Dict[str, Any]:  # type: ignore[override]
        train(req.config, work_dir=req.work_dir)
        return {"status": "started"}

    @app.post("/face-swap/eval")
    def eval_endpoint(req: EvalRequest) -> Dict[str, Any]:  # type: ignore[override]
        return evaluate(req.config, work_dir=req.work_dir)

    @app.post("/face-swap/stream")
    def stream_endpoint(req: StreamRequest) -> Dict[str, Any]:  # type: ignore[override]
        work_dir = Path(req.work_dir) if req.work_dir else Path("work_dirs") / "stream"
        return run_streaming({"config": req.config}, frames=req.frames, work_dir=work_dir)

    @app.get("/reports/{run_id}")
    def get_report(run_id: str) -> Dict[str, Any]:  # type: ignore[override]
        work_dir = Path(run_id)
        metrics_files = list(work_dir.glob("metrics.*.json"))
        graphs = [str(p) for p in work_dir.glob("*.csv")]
        metrics: Dict[str, Any] = {}
        for mf in metrics_files:
            try:
                import json

                metrics[mf.name] = json.loads(mf.read_text())
            except Exception:
                continue
        return {"run_id": run_id, "metrics": metrics, "graphs": graphs}
else:
    app = None
