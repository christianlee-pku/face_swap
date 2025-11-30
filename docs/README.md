# Face Swap System Documentation

## Overview

This project delivers a config-driven face swap system on LFW with reproducible training/eval, standardized metrics/artifacts, and edge-oriented export paths (ONNX/TensorRT/ONNX Runtime). Interfaces include CLI, Python API, and optional REST.

## Features

- Registries for datasets, models, losses, pipelines, runners, exporters.
- Config-driven experiments under `configs/face_swap/`; reproducible `work_dirs/<exp>/` with config snapshots, env hashes, metrics, checkpoints, visuals, README.
- LFW ingestion with detection/alignment (MTCNN fallback), manifests, and deterministic augmentations.
- UNet-based generator; identity loss with ArcFace-style embeddings when available.
- Metrics: identity accuracy, LPIPS/SSIM/PSNR; latency/FPS hooks; human-eval metadata stub.
- Exports: ONNX, TensorRT (trtexec) fallback, ONNX Runtime runner; edge benchmark command.
- REST routes for swap/eval/stream/reports (reports now read stored metrics/graphs).

## Environment & Installation

- Conda env: `face_swap` (Python 3.11).
- Required packages (CI pins): `torch==2.3.1` (CPU wheels via https://download.pytorch.org/whl/cpu), `torchvision==0.18.1`, `torchmetrics==1.4.0.post0`, `lpips==0.1.4`, `facenet-pytorch==2.5.2`, `pyyaml==6.0.3`, `fastapi==0.115.5`, `uvicorn[standard]==0.30.6`, `pytest==9.0.1`, `ruff==0.6.9`, `mypy==1.11.1`, `onnx`.
- Install (example):
  ```bash
  conda create -n face_swap python=3.11 -y
  conda activate face_swap
  python -m pip install -r requirements.txt
  # For CPU wheels: python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
  ```
- Set module path: `export PYTHONPATH=src`

## Dataset Preparation (LFW)

Scripts (set `PYTHONPATH=src`):
- Prepare: `bash scripts/prepare_data.sh`
- Validate: `bash scripts/validate_manifest.sh`
- Train: `bash scripts/train.sh`
- Eval: `bash scripts/eval.sh`
- Infer: `bash scripts/infer.sh`
- Export: `bash scripts/export.sh`
- Benchmark: `bash scripts/benchmark_edge.sh`

Manual equivalents remain available via `interfaces.cli`.

- Behavior: ingests `pairs.txt/pairs_01~pairs_10.txt`, preserves `raw/<person>/<image>.jpg` structure into `processed/`, uses MTCNN alignment, and copies raw images into processed when detection fails to keep manifests relative/complete.

## Configs & Conventions

- Configs under `configs/face_swap/` (baseline, eval, export, ablations).
- Registry keys: `pkg.component.name`.
- Experiment naming: `<task>-<model>-<data>-<id>`; work dirs: `work_dirs/<exp-name>-<timestamp>/`.
- Work dir contents: `config.snapshot.json`, `env.hash`, logs/metrics (JSON/CSV), visuals, checkpoints, README with reproduction commands.

## Training & Evaluation

```bash
python -m interfaces.cli train --config configs/face_swap/baseline.yaml --work-dir work_dirs/<exp>
python -m interfaces.cli eval  --config configs/face_swap/eval.yaml     --work-dir work_dirs/<exp>
```
- Metrics/visuals persisted to work_dir. ArcFace/LPIPS/SSIM/PSNR computed when deps available.

## Inference & Streaming

```bash
python -m interfaces.cli infer --config configs/face_swap/baseline.yaml --sources <src> --targets <tgt> --output-dir work_dirs/infer-samples
```
- Streaming pipeline exists with latency logging; REST `/face-swap/stream` route present (output still placeholder-level; see hardening tasks).

## Export & Edge Benchmark

```bash
python -m interfaces.cli export --config configs/face_swap/baseline.yaml --checkpoint work_dirs/.../checkpoints/best.pth --export-dir work_dirs/exports/baseline
python -m interfaces.cli benchmark-edge --config configs/face_swap/export_edge.yaml --checkpoint work_dirs/.../checkpoints/best.pth --export-dir work_dirs/exports/baseline --target jetson
```
- ONNX via torch.onnx.export (fallback to placeholder if unavailable).
- TensorRT via trtexec (fallback to placeholder).
- ONNX Runtime runner loads session when installed.
- Benchmark logs store measured latency/FPS.

## REST API (FastAPI)

- `/face-swap/batch`, `/face-swap/train`, `/face-swap/eval`, `/face-swap/stream`, `/reports/{run_id}` (reports read metrics/graphs from work_dir).
- Contracts: `specs/001-face-swap-spec/contracts/`.
- Note: Some endpoints still return placeholder outputs until streaming/report artifacts are fully wired.

## Testing & CI

- Tests: `pytest -q --disable-warnings --maxfail=1` (skips for optional deps).
- Lint/type: ruff, mypy (CI enforces).
- CI workflow: `.github/workflows/ci.yml` installs pinned toolchain and runs lint + smoke tests.

## Guides

- Getting Started: `docs/get_started.md`
- Export & Edge: `docs/export_edge.md`
- Streaming & REST: `docs/streaming.md`
- Configs & Conventions: `docs/configs.md`
- Data & Manifests: `docs/data.md`
- Troubleshooting & FAQ: `docs/troubleshooting.md`, `docs/faq.md`
- Validation: `python -m src.data.validate_manifest data/lfw/manifest.json data/lfw/processed`
- Kaggle download: `python -m src.data.preprocess_lfw --download --dataset ashishpatel26/lfw-dataset` (requires Kaggle CLI + credentials)

## Extending Registries

- Add new components in `src/registry/*.py` and register with `Registry.register`.
- Provide example configs under `configs/face_swap/` for new datasets/models/augmentations/runners/exporters.

## Hardening Status (Phase H Tasks)

- Implemented: MTCNN-based alignment, ArcFace embedder, LPIPS/SSIM/PSNR hooks, streaming logging, ONNX/TRT export attempts, ONNX Runtime runner, REST reports reading metrics.
- Remaining to verify/strengthen: Real RetinaFace weights, full streaming outputs, real-world ONNX/TRT validation on target hardware, richer report artifacts.

## References

- Spec: `specs/001-face-swap-spec/spec.md`
- Plan: `specs/001-face-swap-spec/plan.md`
- Tasks: `specs/001-face-swap-spec/tasks.md`
- Quickstart: `specs/001-face-swap-spec/quickstart.md`
- Contracts: `specs/001-face-swap-spec/contracts/`
