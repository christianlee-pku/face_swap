---

description: "Task list template for feature implementation"
---

# Tasks: Face swap system requirements

**Input**: Design documents from `/specs/001-face-swap-spec/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Critical components (registry entries, data pipelines, runners, interfaces) MUST have lint/unit/integration coverage per the constitution. Add extra tests only when requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Constitution Alignment**: Include tasks for config-driven runs (`configs/`, `work_dirs/<exp>/` artifacts), environment pinning, registry additions, metrics/logging/visualization, dataset manifest and augmentation updates, CLI/Python API/REST parity, export/performance for edge targets, and reproducibility checks.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and baseline tooling

- [X] T001 Create project registry/layout scaffolding (configs/, src/registry/, src/data/, src/models/, src/pipelines/, src/runners/, src/exporters/, src/interfaces/, work_dirs/) per plan.md
- [X] T002 Add conda env + lock references in docs (quickstart.md) and document env name `face_swap`
- [X] T003 Configure linting and type checking (ruff/flake8, mypy/pyright) with pyproject/ini files
- [X] T004 Configure pytest with coverage settings and test folder structure (tests/unit, tests/integration, tests/contract)
- [X] T005 Add base logging/metrics utilities scaffolds in src/utils/ (structured logging, JSON-capable)

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure required before user stories

- [X] T006 Implement central registry definitions in src/registry/__init__.py for datasets/models/losses/augmentations/pipelines/runners/exporters
- [X] T007 Add configuration loader and resolver in src/utils/config.py to load `configs/<domain>/<name>.yaml`, apply seeds, and snapshot configs to work_dirs
- [X] T008 Implement work_dir initializer in src/utils/workdir.py to create `work_dirs/<exp-name>-<timestamp>/` with config snapshot and env hash
- [X] T009 Add environment hash capture using conda lock/environment.yml in src/utils/env_info.py
- [X] T010 Implement dataset manifest/versioning helpers in src/data/manifest.py (checksum validation, version bumping)
- [X] T011 Add base runner skeleton in src/runners/base_runner.py with hooks for logging, checkpointing (best + last K), metrics, early stopping
- [X] T012 Implement CLI entrypoint skeleton in src/interfaces/cli.py (train/eval/infer/export commands) wiring to config loader and runner/exporter
- [X] T013 Implement Python API skeleton in src/interfaces/api.py mirroring CLI commands
- [X] T014 Add REST server skeleton (FastAPI) in src/interfaces/rest.py with placeholder routes per contracts/rest.md
- [X] T015 Add initial configs baseline at configs/face_swap/baseline.yaml referencing registry keys (placeholders)
- [X] T016 Add tests: registry import/registration unit tests in tests/unit/registry/test_registry.py
- [X] T017 [P] Add tests: config loader/work_dir/env hash unit tests in tests/unit/utils/test_config_workdir.py
- [X] T018 [P] Add tests: runner skeleton hook behavior unit tests in tests/unit/runners/test_base_runner.py
- [X] T019 [P] Add tests: CLI/API/REST smoke contract tests in tests/contract/test_interfaces.py (placeholder endpoints)

## Phase 3: User Story 1 - ML engineer runs reproducible experiments (Priority: P1) 🎯 MVP

**Goal**: Reproducible config-driven training/evaluation on LFW with experiment artifacts and comparison

**Independent Test**: Run training/eval with a config on sample LFW data and reproduce metrics/logs/checkpoints in a fresh run using the saved config/env.

### Implementation for User Story 1

- [X] T020 [US1] Implement LFW ingestion with RetinaFace detection + 5-point alignment to 112x112 and 80/10/10 split in src/data/lfw_dataset.py with manifest support
- [X] T021 [US1] Implement light augmentations (color jitter, mild blur, horizontal flip) in src/data/transforms.py with deterministic seeds
- [X] T022 [US1] Register datasets/augmentations in registry (src/registry/datasets.py, src/registry/augmentations.py)
- [X] T023 [US1] Implement encoder-decoder/U-Net model in src/models/unet_face_swap.py with export-friendly layers
- [X] T024 [US1] Implement losses (identity perceptual via ArcFace features, adversarial, reconstruction/blending) in src/models/losses.py and register
- [X] T025 [US1] Implement training/eval pipeline in src/pipelines/train_eval.py hooking data, model, losses, metrics (identity accuracy, LPIPS/SSIM/PSNR), checkpointing, visualization
- [X] T026 [US1] Extend runner to wire pipeline, metrics logging, early stopping in src/runners/base_runner.py
- [X] T027 [US1] Add config examples for training/eval at configs/face_swap/baseline.yaml (data, model, loss, optimizer, scheduler, hooks)
- [X] T028 [US1] Add metrics/visualization helpers (plots) in src/utils/metrics_viz.py; persist JSON/CSV + graphs to work_dirs
- [X] T029 [US1] Add comparison utility for two runs (baseline vs new) in src/utils/comparison.py; surfaces metric deltas
- [X] T030 [P] [US1] Add unit tests for datasets/transforms in tests/unit/data/test_lfw_dataset.py
- [X] T031 [P] [US1] Add unit tests for model/loss wiring in tests/unit/models/test_unet_losses.py
- [X] T032 [P] [US1] Add integration test for training/eval pipeline on small subset in tests/integration/test_train_eval.py
- [X] T033 [P] [US1] Add CLI training/eval contract test in tests/contract/test_cli_train_eval.py
- [X] T034 [US1] Update docs for reproducible runs (quickstart.md, README if exists) referencing config/env/work_dir pattern

## Phase 4: User Story 2 - CV researcher evaluates quality and visuals (Priority: P2)

**Goal**: Evaluation with quantitative metrics, qualitative outputs, and regression checks

**Independent Test**: Run eval on held-out split and sample videos, produce metrics (identity accuracy, LPIPS/SSIM/PSNR, latency/FPS) plus graphs and sample frames/clips without modifying training code.

### Implementation for User Story 2

- [X] T035 [US2] Implement evaluation pipeline for images/videos generating metrics and saving sample outputs in src/pipelines/eval_only.py
- [X] T036 [US2] Add latency/FPS measurement hooks for video inference in src/utils/perf.py
- [X] T037 [US2] Add human-rated sample support (small MOS-style set) metadata capture in src/utils/human_eval.py (stub for later collection)
- [X] T038 [US2] Extend visualization/report generation in src/utils/metrics_viz.py to produce comparison graphs and sample galleries
- [X] T039 [US2] Add config for eval-only runs at configs/face_swap/eval.yaml
- [X] T040 [P] [US2] Add integration test for eval-only pipeline on small sample in tests/integration/test_eval_only.py
- [X] T041 [P] [US2] Add contract test for reports endpoint (REST) in tests/contract/test_rest_reports.py
- [X] T042 [US2] Update docs/quickstart with eval/report commands and interpretation guidance

## Phase 5: User Story 3 - Product/embedded engineer deploys low-latency face swap (Priority: P3)

- **Goal**: Export and integrate for batch/real-time use on edge hardware with performance targets.
- **Independent Test**: Export model, run edge benchmark on Jetson profile, and validate ≥30 FPS @ 720p with consistent visual quality.

### Implementation for User Story 3

- [X] T043 [US3] Implement export to ONNX in src/exporters/onnx_exporter.py with dynamic/static axes as needed
- [X] T044 [US3] Implement TensorRT conversion and runner in src/exporters/tensorrt_exporter.py targeting FP16 with fallback
- [X] T045 [US3] Add ONNX Runtime CPU fallback path in src/exporters/onnxruntime_runner.py
- [X] T046 [US3] Add edge benchmark command (CLI) in src/interfaces/cli.py and helper in src/exporters/benchmarks.py for Jetson (≤1.5 GB GPU)
- [X] T047 [US3] Implement streaming/video inference path with frame sampling and temporal smoothing in src/pipelines/streaming.py
- [X] T048 [US3] Wire REST streaming endpoint in src/interfaces/rest.py using shared pipeline
- [X] T049 [US3] Add configs for export/edge benchmark at configs/face_swap/export_edge.yaml
- [X] T050 [P] [US3] Add integration test for ONNX export round-trip in tests/integration/test_export_onnx.py
- [X] T051 [P] [US3] Add integration test for TensorRT pipeline stub (skipped if TRT unavailable) in tests/integration/test_export_trt.py
- [X] T052 [P] [US3] Add contract test for REST streaming endpoint in tests/contract/test_rest_stream.py
- [X] T053 [US3] Update docs (quickstart.md, README if exists) with export/edge steps and performance targets

## Phase N: Polish & Cross-Cutting Concerns

- [X] T054 Add baseline configs for ablations (augmentations/model variants) in configs/face_swap/ablation_*.yaml
- [X] T055 Add reproduction README template in work_dirs generation explaining rerun commands
- [X] T056 Add CI workflow to run lint + unit + selected integration smoke on reference config
- [X] T057 Add dataset update utility in src/data/update_dataset.py and document manifest bump/changelog
- [X] T058 Add additional logging/observability (structured logs, error codes) in src/utils/logging.py
- [X] T059 Add documentation cross-links (spec, plan, quickstart, contracts) in docs/ or README

## Phase H: Hardening & Production Readiness

- [X] T060 Implement RetinaFace-based detection/alignment manifest builder (data/retinaface_align.py) and integrate aligned outputs into src/data/lfw_dataset.py ingestion
- [X] T061 Integrate ArcFace embedding for identity loss/metric using real weights in src/models/arcface.py and wire into src/models/losses.py
- [X] T062 Compute real LPIPS/SSIM/PSNR on model outputs in src/pipelines/train_eval.py and src/pipelines/eval_only.py; persist sample visuals
- [X] T063 Implement streaming/video pipeline with frame sampling and temporal smoothing producing sample outputs in src/pipelines/streaming.py
- [X] T064 Implement torch.onnx.export path and ONNX Runtime validation, plus TensorRT conversion via trtexec in src/exporters/onnx_exporter.py and src/exporters/tensorrt_exporter.py; validate round-trip in src/exporters/onnxruntime_runner.py
- [X] T065 Implement REST reports/streaming responses returning stored metrics/graphs/artifacts in src/interfaces/rest.py
- [X] T066 Add edge benchmark run recording actual 720p@30 FPS results in src/exporters/benchmarks.py and interfaces/cli.py
- [X] T067 Update root README with config/work_dir naming conventions, export/benchmark commands, and add a naming validation/check script if needed
- [X] T068 Add Kaggle LFW download + folder normalization + retry/error handling script; run checksum validation post-download
- [X] T069 Documentation sweep: update install/env/dataset prep (download/validate), training, inference (batch/stream), export/edge, deployment, customization, REST usage across docs/ and quickstart/README
- [X] T070 Add optional dependency entry for RetinaFace and a validation test when installed

## Dependencies & Execution Order

- Foundational (Phase 2) must complete before any user story work.
- Story order: US1 → US2 → US3 (independent once foundation done).
- Exports/edge benchmarking depend on US1 model and US2 evaluation utilities.

## Parallel Opportunities

- Parallel within foundation: T017, T018, T019 after registry/config scaffolds exist.
- US1: T030–T033 can run in parallel with model/pipeline once data/model skeletons exist.
- US2: T040–T041 can run in parallel once eval pipeline stub exists.
- US3: T050–T052 can run in parallel after export stubs exist.

## Implementation Strategy

- MVP: Complete Phases 1–3 (US1) to deliver reproducible training/eval with metrics and artifacts.
- Incremental: Add US2 evaluation/reporting, then US3 export/edge paths.
- Always ensure configs, registry registration, and work_dir artifacts stay in sync across phases.
