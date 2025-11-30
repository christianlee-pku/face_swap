# Troubleshooting

- **ModuleNotFound (torch/onnxruntime/trtexec)**: Install missing deps; for torch CPU use `torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu`. TensorRT requires trtexec on path.
- **FastAPI skips/REST tests skip**: Install `fastapi` and `uvicorn[standard]`.
- **ArcFace/LPIPS metrics missing**: Install `facenet-pytorch`, `lpips`, `torchmetrics`; otherwise metrics fall back to placeholders.
- **Manifest errors**: Regenerate manifest via `build_manifest_from_raw` and ensure paths match `data/lfw/processed`.
- **Export failures**: Check opset/version and input shapes; if trtexec missing, engine file will be placeholder.
- **Streaming outputs empty**: Current streaming is minimal; extend pipeline to write frames and return URLs.
- **CI mypy failures**: Ensure all optional deps installed or add type stubs; CI pins mypy 1.11.1.
