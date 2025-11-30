# FAQ

- **Q: Why are some outputs placeholder?**  
  A: Optional deps (torch, facenet-pytorch, lpips, trtexec) may be missing. Install them to enable full metrics/exports/REST outputs.

- **Q: How do I ensure reproducibility?**  
  A: Use the configs under `configs/face_swap/`, keep seeds set, and reuse the same manifest/splits. Work dirs keep `config.yaml`/`config.py`, metrics, and checkpoints.

- **Q: How to benchmark edge FPS?**  
  A: Run `python -m interfaces.cli benchmark-edge ...` with export config/checkpoint; replace placeholder latency with real device run and record in `benchmark.json`.

- **Q: Can I change augmentations/model easily?**  
  A: Yes—modify configs to reference registered components; avoid code changes where possible.

- **Q: How do I add a new dataset/model?**  
  A: Register in `src/registry`, add defaults/configs, add tests, and ensure manifests/splits documented.
