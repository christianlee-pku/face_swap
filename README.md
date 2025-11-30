# Face Swap System (LFW)

Config-driven Python/PyTorch project for face swapping on LFW with reproducible training/eval, metrics, and edge exports.

## Environment

- Conda env: `face_swap`
- Python 3.11, PyTorch (CPU by default), torchvision/torchmetrics, PyYAML, FastAPI (optional), lpips/ArcFace deps as needed.

## Key Commands (config-driven)

- Data prep: `bash scripts/prepare_data.sh` (uses `configs/face_swap/data_prepare.yaml`)
- Validate manifest: `bash scripts/validate_manifest.sh`
- Train: `bash scripts/train.sh` (uses `configs/face_swap/baseline.yaml`)
- Eval: `bash scripts/eval.sh` (uses `configs/face_swap/eval.yaml`)
- Infer: `bash scripts/infer.sh` (uses `configs/face_swap/infer.yaml`)
- ONNX export: `bash scripts/export.sh` (uses `configs/face_swap/export.yaml`)
- TensorRT export: `bash scripts/trt.sh` (uses `configs/face_swap/trt.yaml`; requires trtexec)
- Edge benchmark: `bash scripts/benchmark_edge.sh` (uses `configs/face_swap/export.yaml`)

## Project Structure

- `configs/face_swap/` — experiment configs (baseline, eval, export, ablations)
- `src/registry/` — registries
- `src/data/` — LFW dataset, manifest/update utilities, alignment stub
- `src/models/` — UNet, losses, ArcFace stub
- `src/pipelines/` — train/eval/streaming
- `src/exporters/` — ONNX/TensorRT/ONNX Runtime stubs, benchmarks
- `src/interfaces/` — CLI, Python API, REST
- `tests/` — unit/integration/contract tests
- `work_dirs/` — run artifacts (configs, metrics, checkpoints)

## Docs

- Main docs: `docs/README.md`
- Data guide: `docs/data.md` (includes Kaggle download command and validation)
- Dependencies (CI installs; see requirements.txt): torch CPU, torchvision, torchmetrics, lpips, facenet-pytorch, fastapi/uvicorn, pytest, ruff, mypy
- Data prep (config-driven): `bash scripts/prepare_data.sh` (ingests `pairs.txt/pairs_01~pairs_10.txt`, MTCNN alignment to processed folder; falls back to copying raw if detection fails); validate with `bash scripts/validate_manifest.sh`
- Pipeline scripts (set `PYTHONPATH=src`):
  - Prepare data: `bash scripts/prepare_data.sh`
  - Validate manifest: `bash scripts/validate_manifest.sh`
  - Train/Eval: `bash scripts/train.sh`, `bash scripts/eval.sh`
  - Infer: `bash scripts/infer.sh`
  - Export/Benchmark: `bash scripts/export.sh`, `bash scripts/benchmark_edge.sh`
