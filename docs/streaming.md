# Streaming & REST

## Endpoints

- `/face-swap/stream`: accepts frames list, config, optional work_dir; returns latency/FPS and processed frame count.
- `/reports/{run_id}`: reads metrics/graphs from work_dir and returns JSON links.
- Other routes: `/face-swap/batch`, `/face-swap/train`, `/face-swap/eval`.

## Streaming Pipeline

- Implementation: `src/pipelines/streaming.py` (logs frame count, latency/FPS, writes stream.log).
- REST integration: `src/interfaces/rest.py` uses `run_streaming`.
- Status: functional placeholder; integrate real frame sampling, temporal smoothing, and output writing for production.

## Usage

Start FastAPI (example with uvicorn):
```bash
uvicorn src.interfaces.rest:app --host 0.0.0.0 --port 8000
```

POST streaming:
```bash
curl -X POST http://localhost:8000/face-swap/stream \
  -H "Content-Type: application/json" \
  -d '{"config": "configs/face_swap/baseline.yaml", "frames": ["frame1.png", "frame2.png"]}'
```

Reports:
```bash
curl http://localhost:8000/reports/work_dirs/lfw-unet-baseline-<ts>
```

## Hardening Steps

- Add real video frame ingestion (decode, resize, align) and temporal smoothing.
- Write sample outputs/frames to work_dir and return URLs in REST responses.
- Add authentication/quotas if exposed beyond internal use.
- Benchmark streaming FPS/latency on target edge hardware.***
