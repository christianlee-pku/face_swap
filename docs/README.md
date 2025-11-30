# Face Swap System Documentation

## Overview

This project delivers a config-driven face swap system on LFW with reproducible training/eval, standardized metrics/artifacts, and edge-oriented export paths (ONNX/TensorRT/ONNX Runtime). Interfaces include CLI, Python API, and optional REST.

## Features

- Registries for datasets, models, losses, pipelines, runners, exporters.
- Config-driven experiments under `configs/face_swap/`; reproducible `work_dirs/<exp>/` with configs, metrics, checkpoints.
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
- Train: `bash scripts/train.sh` (configs/face_swap/baseline.yaml)
- Eval: `bash scripts/eval.sh` (configs/face_swap/eval.yaml)
- Infer: `bash scripts/infer.sh` (configs/face_swap/infer.yaml)
- Export: `bash scripts/export.sh` (configs/face_swap/export.yaml)
- TensorRT: `bash scripts/trt.sh` (configs/face_swap/trt.yaml; needs trtexec)
- Benchmark: `bash scripts/benchmark_edge.sh` (configs/face_swap/export.yaml)

Manual equivalents remain available via `interfaces.cli`.

- Behavior: ingests `pairs.txt/pairs_01~pairs_10.txt`, preserves `raw/<person>/<image>.jpg` structure into `processed/`, uses MTCNN alignment, and copies raw images into processed when detection fails to keep manifests relative/complete.

## Configs & Conventions

- Configs under `configs/face_swap/` (baseline, eval, export, ablations).
- Registry keys: `pkg.component.name`.
- Experiment naming: `<task>-<model>-<data>-<id>`; work dirs: `work_dirs/<exp-name>-<timestamp>/`.
- Work dir contents: `config.yaml`, `config.py`, logs, metrics (`metrics.train.json`), checkpoints.

## Training & Evaluation

```bash
python -m interfaces.cli train --config configs/face_swap/baseline.yaml --work-dir work_dirs/<exp>
python -m interfaces.cli eval  --config configs/face_swap/eval.yaml     --work-dir work_dirs/<exp>
```
- Metrics/visuals persisted to work_dir. ArcFace/LPIPS/SSIM/PSNR computed when deps available.

## Inference

- Use `bash scripts/infer.sh` (sources/targets/output_dir from `configs/face_swap/infer.yaml`).

## Export & Edge Benchmark

- ONNX export: `bash scripts/export.sh` (uses `configs/face_swap/export.yaml`)
- TensorRT export: `bash scripts/trt.sh` (uses `configs/face_swap/trt.yaml`; requires trtexec)
- Edge benchmark: `bash scripts/benchmark_edge.sh` (uses `configs/face_swap/export.yaml`)

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
- Validation: `bash scripts/validate_manifest.sh`
- Kaggle download: `bash scripts/prepare_data.sh` (requires Kaggle CLI + credentials)

## Extending Registries

- Add new components in `src/registry/*.py` and register with `Registry.register`.
- Provide example configs under `configs/face_swap/` for new datasets/models/augmentations/runners/exporters.

## Hardening Status (Phase H Tasks)

- Implemented: MTCNN-based alignment, ArcFace embedder, LPIPS/SSIM/PSNR hooks, streaming logging, ONNX/TRT export attempts, ONNX Runtime runner, REST reports reading metrics.
- Remaining to verify/strengthen: Real RetinaFace weights, full streaming outputs, real-world ONNX/TRT validation on target hardware, richer report artifacts.

## References

- Getting Started: `docs/get_started.md`
- Export & Edge: `docs/export_edge.md`
- Streaming & REST: `docs/streaming.md`
- Configs & Conventions: `docs/configs.md`
- Data & Manifests: `docs/data.md`
- Troubleshooting & FAQ: `docs/troubleshooting.md`, `docs/faq.md`
