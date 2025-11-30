# Export & Edge Deployment

## Goals

- Produce ONNX and TensorRT artifacts for Jetson-class targets (≤1.5 GB GPU, FP16 preferred).
- Validate ONNX Runtime inference.
- Benchmark latency/FPS at 720p with recorded results in `work_dirs/exports/<exp>/`.

## Prerequisites

- Install torch (CPU ok for export), onnxruntime, and TensorRT tooling (`trtexec`) on the export host.
- Checkpoint from training: `work_dirs/.../checkpoints/best.pth`.

## ONNX Export

```bash
python -m interfaces.cli export \
  --config configs/face_swap/baseline.yaml \
  --checkpoint work_dirs/.../checkpoints/best.pth \
  --export-dir work_dirs/exports/baseline
```
- Uses torch.onnx.export when available; falls back to placeholder if missing deps.
- Output: `model.onnx` in export_dir.

## TensorRT Conversion

- Conversion invoked via `trtexec` in `exporters/tensorrt_exporter.py` (FP16 by default).
- Output: `model.fp16.engine` (placeholder if trtexec missing).

## ONNX Runtime Validation

- `exporters/onnxruntime_runner.py` loads the ONNX and runs inference if onnxruntime is installed.
- Provide an input dict with key `input` matching export shapes.

## Edge Benchmark

```bash
python -m interfaces.cli benchmark-edge \
  --config configs/face_swap/export_edge.yaml \
  --checkpoint work_dirs/.../checkpoints/best.pth \
  --export-dir work_dirs/exports/baseline \
  --target jetson
```
- Records latency/FPS to `benchmark.json`. Current latency/FPS uses perf hooks; replace with real device measurements.

## Packaging

- Include: `model.onnx`, `model.fp16.engine` (if built), `benchmark.json`, sample commands, config snapshot, env hash.
- Document device assumptions (Jetson Orin/Xavier, FP16).

## Hardening Notes

- Replace placeholder metrics with real device benchmarks.
- Validate ONNX/engine outputs against reference frames.
- Add quantization/int8 if needed (not implemented).***
