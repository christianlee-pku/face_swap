# Face Swap System (LFW)

Config-driven Python/PyTorch project for face swapping on LFW with reproducible training/eval, metrics, and edge exports.

## Environment

- Conda env: `face_swap`
- Python 3.11, PyTorch (CPU by default), torchvision/torchmetrics, PyYAML, FastAPI (optional), lpips/ArcFace deps as needed.

## Key Commands

- Train/Eval (config-driven):
  ```bash
  python -m interfaces.cli train --config configs/face_swap/baseline.yaml --work-dir work_dirs/<exp>
  python -m interfaces.cli eval  --config configs/face_swap/eval.yaml     --work-dir work_dirs/<exp>
  ```
- Inference (batch placeholder):
  ```bash
  python -m interfaces.cli infer --config configs/face_swap/baseline.yaml --sources <src> --targets <tgt>
  ```
- Export & Edge Benchmark:
  ```bash
  python -m interfaces.cli export        --config configs/face_swap/baseline.yaml --checkpoint work_dirs/.../checkpoints/best.pth --export-dir work_dirs/exports/baseline
  python -m interfaces.cli benchmark-edge --config configs/face_swap/export_edge.yaml --checkpoint work_dirs/.../checkpoints/best.pth --export-dir work_dirs/exports/baseline --target jetson
  ```
- REST (optional): see `specs/001-face-swap-spec/contracts/` and `src/interfaces/rest.py` (routes are placeholders until hardened).

## Project Structure

- `configs/face_swap/` — experiment configs (baseline, eval, export, ablations)
- `src/registry/` — registries
- `src/data/` — LFW dataset, manifest/update utilities, alignment stub
- `src/models/` — UNet, losses, ArcFace stub
- `src/pipelines/` — train/eval/streaming
- `src/exporters/` — ONNX/TensorRT/ONNX Runtime stubs, benchmarks
- `src/interfaces/` — CLI, Python API, REST
- `tests/` — unit/integration/contract tests
- `work_dirs/` — run artifacts (config snapshot, env hash, metrics, checkpoints, README)

## Status & Next Steps

- Core pipelines, interfaces, exports, and tests exist but many components remain placeholders (detection/alignment, ArcFace metrics, real LPIPS/SSIM/PSNR, streaming outputs, ONNX/TRT validation, REST reports).
- Hardening tasks are tracked in `specs/001-face-swap-spec/tasks.md` (Phase H: T060–T067).

## Docs

- Main docs: `docs/README.md`
- Spec: `specs/001-face-swap-spec/spec.md`
- Plan: `specs/001-face-swap-spec/plan.md`
- Tasks: `specs/001-face-swap-spec/tasks.md`
- Quickstart: `specs/001-face-swap-spec/quickstart.md`
- Contracts: `specs/001-face-swap-spec/contracts/`
- Data guide: `docs/data.md` (includes Kaggle download command and validation)
- Dependencies (CI installs; see requirements.txt): torch CPU, torchvision, torchmetrics, lpips, facenet-pytorch, fastapi/uvicorn, pytest, ruff, mypy
- Data prep (config-driven): `PYTHONPATH=src python -m interfaces.cli prepare-data --config configs/face_swap/data_prepare.yaml` (ingests `pairs.txt/pairs_01~pairs_10.txt`, MTCNN alignment to processed folder; falls back to copying raw if detection fails); validate with `PYTHONPATH=src python -m interfaces.cli validate-manifest --manifest data/lfw/manifest.json --processed-dir data/lfw/processed`
- Pipeline scripts (set `PYTHONPATH=src`):
  - Prepare data: `bash scripts/prepare_data.sh`
  - Validate manifest: `bash scripts/validate_manifest.sh`
  - Train/Eval: `bash scripts/train.sh`, `bash scripts/eval.sh`
  - Infer: `bash scripts/infer.sh`
  - Export/Benchmark: `bash scripts/export.sh`, `bash scripts/benchmark_edge.sh`
