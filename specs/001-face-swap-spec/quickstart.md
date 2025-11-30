# Quickstart: Face swap system requirements

**Branch**: `001-face-swap-spec`  
**Date**: 2025-11-27  
**Spec**: `/Users/christian/Documents/projects/face_generation/specs/001-face-swap-spec/spec.md`

## Prerequisites

- Conda env: `face_swap` (Python 3.11, PyTorch, registry utils, RetinaFace deps, ONNX/TensorRT, FastAPI optional).
- Clone repo and ensure `configs/` and `work_dirs/` are writable.
- Set module path: `export PYTHONPATH=src`

## Dataset Preparation (LFW)

1. Download/prepare via config-driven CLI:
   ```bash
   PYTHONPATH=src python -m interfaces.cli prepare-data --config configs/face_swap/data_prepare.yaml
   ```
   - Ingests `pairs.txt` and `pairs_01~pairs_10.txt`, preserves `raw/<person>/<image>.jpg` layout into `processed/`, uses MTCNN alignment, and copies raw images into processed when detection fails to keep manifests consistent.
2. Validate manifest/checksums:
   ```bash
   PYTHONPATH=src python -m interfaces.cli validate-manifest --manifest data/lfw/manifest.json --processed-dir data/lfw/processed
   ```

## Training & Evaluation

```bash
# Activate env
conda activate face_swap

# Train/eval with a config
PYTHONPATH=src python -m interfaces.cli train \
  --config configs/face_swap/baseline.yaml \
  --work-dir work_dirs/lfw-unet-baseline-001-$(date +%s)
```

Outputs in `work_dirs/<exp-name>-<timestamp>/`: config snapshot, env hash, logs, metrics (JSON/CSV), graphs, checkpoints (best + last K), sample outputs, reproduction README.

## Inference (Batch & Streaming)

```bash
python -m interfaces.cli infer \
  --config configs/face_swap/baseline.yaml \
  --sources path/to/source.jpg,path/to/source2.jpg \
  --targets path/to/target.jpg \
  --output-dir work_dirs/infer-samples
```

REST (optional): start FastAPI server (to be implemented) and POST paths/URLs or base64 payloads; responses include artifacts and JSON metadata with metric URLs.

## Streaming (Placeholder)

- REST: POST `/face-swap/stream` with frames list; current response is placeholder until streaming pipeline is hardened.

## Export & Edge Benchmark

```bash
python -m interfaces.cli export \
  --config configs/face_swap/baseline.yaml \
  --checkpoint work_dirs/.../checkpoints/best.pth \
  --export-dir work_dirs/exports/baseline

python -m interfaces.cli benchmark-edge \
  --config configs/face_swap/export_edge.yaml \
  --checkpoint work_dirs/.../checkpoints/best.pth \
  --export-dir work_dirs/exports/baseline \
  --target jetson
```

Exports: ONNX + TensorRT (FP16 preferred) for Jetson-class (≤1.5 GB GPU mem), with CPU fallback via ONNX Runtime; benchmark logs stored under `work_dirs/exports/<exp>/`.

## Reproducibility

- Use provided `environment.yml` and lockfile; update hashes on dependency changes.
- Set seeds in config for loaders/augmentations/model init.
- Keep experiment naming as `<task>-<model>-<data>-<id>`; use `work_dirs/<exp-name>-<timestamp>/`.

## Evaluation & Reports (US2)

- Eval-only run producing metrics/graphs/sample outputs:

```bash
python -m interfaces.cli eval \
  --config configs/face_swap/eval.yaml \
  --work-dir work_dirs/lfw-unet-eval-001-$(date +%s)
```

- REST (optional): GET `/reports/{run_id}` (placeholder) to fetch metrics/graph links once stored.

## Notes

- Many components are placeholders (detection/alignment, ArcFace metrics, real LPIPS/SSIM/PSNR, streaming outputs, ONNX/TRT validation). See Phase H tasks in `tasks.md` for hardening steps.

## References

- Spec: specs/001-face-swap-spec/spec.md
- Plan: specs/001-face-swap-spec/plan.md
- Tasks: specs/001-face-swap-spec/tasks.md
- Contracts: specs/001-face-swap-spec/contracts/
