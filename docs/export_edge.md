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
bash scripts/export.sh                # uses configs/face_swap/export.yaml
```
- Output: `model.onnx` in export_dir (fails if torch/onnx missing or checkpoint mismatch).

## TensorRT Conversion

- Conversion invoked via `trtexec` (config: configs/face_swap/trt.yaml, script: bash scripts/trt.sh).
- Output: `model.trt` (fails if trtexec missing).

## ONNX Runtime Validation

- `exporters/onnxruntime_runner.py` loads the ONNX and runs inference if onnxruntime is installed.
- Provide an input dict with key `input` matching export shapes.

## Edge Benchmark

```bash
bash scripts/benchmark_edge.sh    # uses configs/face_swap/export.yaml by default
```
- Records latency/FPS to `benchmark.json`. Current latency/FPS uses perf hooks; replace with real device measurements.

## Packaging

- Include: `model.onnx`, `model.trt` (if built), `benchmark.json`, sample commands.
- Document device assumptions (Jetson Orin/Xavier, FP16).

## Hardening Notes

- Replace placeholder metrics with real device benchmarks.
- Validate ONNX/engine outputs against reference frames.
- Add quantization/int8 if needed (not implemented).***
