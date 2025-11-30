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

1) Download/prepare via config:
   ```bash
   # Requires Kaggle CLI with KAGGLE_USERNAME/KAGGLE_KEY set
   bash scripts/prepare_data.sh
   ```
   - Ingests `pairs.txt` and `pairs_01~pairs_10.txt`, preserves folder structure, uses MTCNN alignment; if no face is detected, the raw image is copied into `data/lfw/processed`.
2) Validate manifest/checksums:
   ```bash
   bash scripts/validate_manifest.sh
   ```

## Train & Eval

- Train: `bash scripts/train.sh` (baseline config)
- Eval: `bash scripts/eval.sh` (eval config)
- Outputs: `config.yaml`, `config.py`, logs, `metrics.train.json`, checkpoints.

## Inference (Batch)

- `bash scripts/infer.sh` (uses `configs/face_swap/infer.yaml` for sources/targets/output_dir/checkpoint)

## Export & Edge Benchmark

- ONNX: `bash scripts/export.sh` (uses `configs/face_swap/export.yaml`)
- TensorRT: `bash scripts/trt.sh` (uses `configs/face_swap/trt.yaml`; requires trtexec)
- Benchmark: `bash scripts/benchmark_edge.sh` (uses `configs/face_swap/export.yaml`)

## REST (Optional)

- Routes: `/face-swap/batch`, `/face-swap/train`, `/face-swap/eval`, `/face-swap/stream`, `/reports/{run_id}` (placeholder status; see `src/interfaces/rest.py`).

## Notes

- Some components still rely on optional deps; install torch/torchvision/torchmetrics/lpips/facenet-pytorch for full metrics.
- Streaming and ONNX/TRT behavior depend on runtime availability (trtexec, onnxruntime).
