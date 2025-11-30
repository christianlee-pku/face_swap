# Implementation Plan: Face swap system requirements

**Branch**: `001-face-swap-spec` | **Date**: 2025-11-27 | **Spec**: `/Users/christian/Documents/projects/face_generation/specs/001-face-swap-spec/spec.md`
**Input**: Feature specification from `/specs/001-face-swap-spec/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command or equivalent automation. Keep entries aligned with the constitution.

## Summary

Config-driven Python/PyTorch face swap system on LFW with automated dataset pipeline (Kaggle download, normalize folders, detect/align via MTCNN, 112x112 crop/resize, checksums, 80/10/10 splits, deterministic augmentations); UNet-based model with ArcFace/LPIPS/SSIM/PSNR metrics; reproducible runs in `work_dirs`; CLI/Python/REST interfaces for batch/streaming (REST internal/no-auth for MVP); exports to ONNX/TensorRT/ORT for edge targets with docs covering install/env/dataset prep/training/inference/deployment/customization.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11 (conda env `face_swap`)  
**Primary Dependencies**: PyTorch, torchvision/torchmetrics, lpips, facenet-pytorch (ArcFace/MTCNN), PyYAML, FastAPI (optional REST), ONNX + TensorRT + ONNX Runtime, click/typer for CLI  
**Storage**: Filesystem (datasets, manifests, configs, work_dirs artifacts)  
**Testing**: pytest + coverage; lint/type via ruff + mypy (enforced in CI)  
**Target Platform**: Training on GPU workstation; inference on Jetson-class edge (Orin/Xavier) + CPU fallback  
**Project Type**: Single Python project with modular registries  
**Performance Goals**: ≥30 FPS @ 720p on target edge device; reproducible metrics/graphs; stable video blending  
**Constraints**: ≤1.5 GB GPU mem on edge; deterministic seeds; pinned env/lock; batch/streaming support; REST optional  
**Scale/Scope**: LFW (~13k images) + extensions; batch/offline + real-time video flows

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Config-driven reproducibility: configs in `configs/face_swap/`, env pinned, seeds, `work_dirs/<exp>/` with config snapshot/env hash/logs/metrics/checkpoints/visuals/README.
- Registry architecture: datasets/models/losses/augmentations/pipelines/runners/exporters registered via central registry; no ad-hoc wiring.
- Standardized training/eval: runner hooks for logging, checkpointing (best+last K), metrics (identity accuracy, LPIPS/SSIM/PSNR, latency), early stop, visualization, regression checks.
- Data integrity/augmentation: LFW download/align/crop/resize + checksums/manifests; deterministic splits/augs; update utilities with versioned manifests.
- Interfaces and deployment: CLI/Python API/REST share pipelines; batch + streaming; edge target 720p @ ≥30 FPS; export path ONNX → TensorRT/ONNX Runtime.
- Quality gates: lint/type (ruff+mypy), pytest unit/integration/contract, CI artifact layout in `work_dirs/<exp>/`, reproduction check for reference config.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
configs/
└── face_swap/                 # YAML configs (baseline, eval, export, ablations)

src/
├── registry/                  # registries for datasets, models, losses, augmentations, pipelines, runners, exporters
├── data/                      # LFW ingestion, alignment, manifests, update utilities, transforms
├── models/                    # UNet, ArcFace embedder, losses
├── pipelines/                 # train/eval/streaming pipelines
├── runners/                   # standardized runner
├── exporters/                 # ONNX/TensorRT/ONNX Runtime, benchmarks
├── interfaces/                # CLI, Python API, REST
└── utils/                     # logging, metrics, perf, comparison

tests/
├── unit/
├── integration/
└── contract/

docs/                          # project docs (get_started, export_edge, streaming, configs, data, FAQ)
work_dirs/                     # experiment artifacts
```

**Structure Decision**: Single Python project with registries and config-driven runs.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
