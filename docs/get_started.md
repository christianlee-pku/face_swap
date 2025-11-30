# Getting Started

## Environment

- Conda env: `face_swap`
- Python 3.11 with pinned deps (torch CPU, torchmetrics, lpips, facenet-pytorch, pyyaml, fastapi, uvicorn, onnx).
- Install:
  ```bash
  conda create -n face_swap python=3.11 -y
  conda activate face_swap
  python -m pip install -r requirements.txt
  # For CPU: python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
  ```

## Dataset (LFW)

1) Download/prepare LFW via config or flags:
   ```bash
   # Requires Kaggle CLI with KAGGLE_USERNAME/KAGGLE_KEY set
   PYTHONPATH=src python -m interfaces.cli prepare-data --config configs/face_swap/data_prepare.yaml
   ```
   - Ingests `pairs.txt` and `pairs_01~pairs_10.txt`, preserves folder structure, and uses MTCNN alignment; if no face is detected, the raw image is copied into `data/lfw/processed` to keep manifests consistent.
2) Align and build manifest (already included when using --download; rerun if needed):
   ```bash
   PYTHONPATH=src python -m interfaces.cli prepare-data --config configs/face_swap/data_prepare.yaml
   ```
3) Validate manifest/checksums:
   ```bash
   PYTHONPATH=src python -m interfaces.cli validate-manifest --manifest data/lfw/manifest.json --processed-dir data/lfw/processed
   ```

## Train & Eval

```bash
python -m interfaces.cli train --config configs/face_swap/baseline.yaml --work-dir work_dirs/lfw-unet-baseline-$(date +%s)
python -m interfaces.cli eval  --config configs/face_swap/eval.yaml     --work-dir work_dirs/lfw-unet-eval-$(date +%s)
```
- Outputs: `config.snapshot.json`, `env.hash`, metrics JSON/CSV, checkpoints, sample visuals (when available), README.

## Inference (Batch)

```bash
python -m interfaces.cli infer \
  --config configs/face_swap/baseline.yaml \
  --sources path/to/source.jpg \
  --targets path/to/target.jpg \
  --output-dir work_dirs/infer-samples
```

## Export & Edge Benchmark

See `docs/export_edge.md` for full flow; quick commands:
```bash
python -m interfaces.cli export --config configs/face_swap/baseline.yaml --checkpoint work_dirs/.../checkpoints/best.pth --export-dir work_dirs/exports/baseline
python -m interfaces.cli benchmark-edge --config configs/face_swap/export_edge.yaml --checkpoint work_dirs/.../checkpoints/best.pth --export-dir work_dirs/exports/baseline --target jetson
```

## REST (Optional)

- Routes: `/face-swap/batch`, `/face-swap/train`, `/face-swap/eval`, `/face-swap/stream`, `/reports/{run_id}`.
- Contracts: `specs/001-face-swap-spec/contracts/`.

## Notes

- Some components still rely on optional deps; install torch/torchvision/torchmetrics/lpips/facenet-pytorch for full metrics.
- Streaming and ONNX/TRT behavior depend on runtime availability (trtexec, onnxruntime).
