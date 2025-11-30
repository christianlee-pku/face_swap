# Feature Specification: Face swap system requirements

**Feature Branch**: `001-face-swap-spec`  
**Created**: 2025-11-27  
**Status**: Draft  
**Input**: User description: "Specify the detailed requirements and user stories for an AI-powered face swap system that takes one or more source faces and seamlessly swaps them into target images or videos while preserving expression, skin tone, and natural blending; supports training on the LFW dataset, configuration-driven experiments, and reproducible evaluation with accuracy/quality metrics and graphs; provides both CLI/Python/REST interfaces for batch and real-time inference; includes utilities to incrementally update/extend the dataset; and offers a clear deployment path to embedded/edge devices for low-latency video face swapping. Capture user stories for ML engineers (experimenting with models/configs), CV researchers (evaluating metrics and visual results), and product/embedded engineers (exporting models, integrating APIs, and running on constrained hardware) within a modular, registry-based project structure."

## Clarifications

### Session 2025-11-27

- Q: Preferred model architecture and loss setup for identity/expression/skin tone preservation? → A: Encoder-decoder/U-Net with identity perceptual, adversarial, and reconstruction/blending losses.
- Q: Target latency/resolution for real-time edge video? → A: 720p @ ≥30 FPS on target edge device.
- Q: LFW preprocessing (detection/alignment/splits/augs)? → A: RetinaFace detection, 5-point alignment to 112x112, 80/10/10 split, light color/blur/flip augmentations.
- Q: API I/O contracts for CLI/Python/REST? → A: Paths/URLs (REST also base64), batch lists supported, return artifacts plus JSON metadata (metrics, URLs).
- Q: Embedded/edge deployment constraints (hardware/memory/runtime/export)? → A: NVIDIA Jetson-class (Orin/Xavier), TensorRT/ONNX, ≤1.5 GB GPU mem, FP16 preferred, CPU fallback via ONNX Runtime, include edge benchmark command.
- Q: How to define/measure quality and accuracy (metrics + human eval + graphs)? → A: Objective metrics (identity accuracy via ArcFace, LPIPS/SSIM/PSNR, latency/FPS) plus small human-rated sample (MOS-style) and trend/comparison graphs.
- Q: Config/registry/work_dir and experiment naming conventions? → A: `configs/<domain>/<name>.yaml`, registry keys `pkg.component.name`, `work_dirs/<exp-name>-<timestamp>/`, experiment names `<task>-<model>-<data>-<id>`.

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - ML engineer runs reproducible experiments (Priority: P1)

An ML engineer wants to automatically prepare LFW (download, normalize directories, detect/align faces, crop/resize, split, augment), then design, run, and compare face swap experiments using declarative configs and reproducible environments.

**Why this priority**: Reproducible experimentation is foundational for model quality and governs all downstream usage.

**Independent Test**: One command prepares LFW (download → align → split → augment) and another runs training/eval from CLI; the same metrics/logs/checkpoints reproduce on a second machine using saved config/env.

**Acceptance Scenarios**:

1. **Given** a pinned environment lock and a config referencing registry components, **When** the engineer runs training/eval via CLI, **Then** the system produces `work_dirs/<exp>/` with config snapshot, logs, metrics, visualizations, and checkpoints that can be replayed for identical results.
2. **Given** two experiment configs with different augmentations, **When** the engineer triggers comparison, **Then** the system outputs side-by-side metric tables/graphs and highlights best-performing checkpoints.

---

### User Story 2 - CV researcher evaluates quality and visuals (Priority: P2)

A computer vision researcher wants to inspect quantitative metrics and qualitative outputs for face swaps on images and videos, ensuring identity preservation, natural blending, and stable motion.

**Why this priority**: Objective and visual evaluation validates the model’s suitability before productization.

**Independent Test**: Run evaluation on a held-out LFW split and sample videos, then review generated reports/graphs and visual overlays without modifying training pipelines.

**Acceptance Scenarios**:

1. **Given** evaluation inputs (target media and source faces) and a trained checkpoint, **When** the researcher runs eval, **Then** the system reports identity accuracy, LPIPS/SSIM/PSNR, latency/FPS, and provides graphs plus sample frames/clips.
2. **Given** a regression baseline, **When** the researcher compares current results, **Then** the system flags any metric drops and points to corresponding checkpoints/visuals.

---

### User Story 3 - Product/embedded engineer deploys low-latency face swap (Priority: P3)

A product or embedded engineer wants to export and integrate the model for batch and real-time use via CLI, Python API, or REST, and run on constrained edge hardware with acceptable latency.

**Why this priority**: Deployment quality determines real-world viability and user experience.

**Independent Test**: Export a trained model, integrate via provided API on a target device profile, and validate latency/FPS and visual stability on sample videos.

**Acceptance Scenarios**:

1. **Given** a trained checkpoint, **When** the engineer exports to an optimized artifact and runs the edge benchmark command, **Then** the system reports FPS/latency meeting target budgets with consistent visual quality.
2. **Given** batch images and a REST endpoint, **When** the engineer submits a job, **Then** the system processes in-order with preserved expressions/tones and returns URLs or artifacts without manual tuning.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- No detectable source face or low-quality source (blur/occlusion) – system must surface actionable errors and skip unsafe swaps.
- Target video frames with rapid motion or heavy lighting shifts – system must stabilize alignment and avoid temporal artifacts.
- Mismatched resolutions/aspect ratios between source and target media – system must normalize inputs and maintain proportions.
- Dataset update adds or removes identities – system must version manifests and prevent silent evaluation drift.
- Inference under constrained hardware where target FPS cannot be met – system must report shortfall and suggest configuration adjustments.
- LFW download or checksum failures – system must abort with clear retry guidance and not generate partial manifests.
- REST exposure beyond internal networks – must revisit auth and harden before public access.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Constitution Alignment *(must be completed for this project)*

- **Reproducibility/Config**: All experiments defined via versioned configs under `configs/`, using pinned conda env + lock; `work_dirs/<exp>/` must store config snapshot, env lock hash, logs, metrics, checkpoints, and visuals; seeds set across loaders/augmentations/model init.
- **Registry Additions**: Datasets (LFW + extensions), data pipelines, models, augmentations, schedulers, exporters, and runners registered with documented contracts and default configs; no ad-hoc wiring.
- **Metrics & Reporting**: Required metrics: identity preservation accuracy on LFW splits, LPIPS/SSIM/PSNR for swap quality, latency/FPS for video; outputs include JSON/CSV metrics, comparison graphs, and sample frames/clips.
- **Data & Augmentation**: LFW prep with checksums/manifests, declared splits, deterministic augmentations that preserve identity; dataset update utilities version manifests and changelogs.
- **Interfaces & Deployment**: CLI, Python API, and REST share the same pipelines and I/O schema for batch and streaming; exports support optimized artifacts for edge devices; performance budgets documented for target devices.

### Functional Requirements

- **FR-001**: Provide versioned experiment configs (YAML) that fully define data, model, augmentation, training, evaluation, and export parameters without hidden defaults.
- **FR-002**: Enforce pinned environments (conda file + lock) and capture seeds/config snapshots, dataset manifests, and git commit for every run in `work_dirs/<exp>/`.
- **FR-003**: Support registry-based composition for datasets, pipelines, models, losses, schedulers, exporters, and runners with discoverable names and documented contracts.
- **FR-004**: Train and evaluate on LFW (and extensions) with checksum-validated ingestion and versioned manifests for any dataset updates; default model is an encoder-decoder/U-Net generator with identity perceptual loss (e.g., ArcFace features), adversarial loss, and reconstruction/blending losses to preserve identity, expression, and skin tone.
- **FR-005**: Run training/eval from CLI and Python API; optional REST layer must reuse the same pipelines and configs.
- **FR-006**: Compute and persist required metrics (identity accuracy via ArcFace features, LPIPS/SSIM/PSNR, latency/FPS), run small human-rated samples (MOS-style) for visual quality, and render trend/comparison graphs and sample outputs.
- **FR-007**: Provide experiment comparison reports highlighting deltas versus a baseline checkpoint for regressions or improvements.
- **FR-008**: Offer batch and streaming inference for images and videos with controls for face selection, blending strength, and frame sampling; inputs accept file/URL paths (REST also supports base64), batches as lists, and outputs include artifacts plus JSON metadata with metrics and URLs.
- **FR-009**: Export optimized artifacts suitable for edge/embedded deployment and include commands to benchmark FPS/latency on target profiles; target NVIDIA Jetson-class (Orin/Xavier) with ≤1.5 GB GPU memory using TensorRT/ONNX, FP16 preferred, with CPU fallback via ONNX Runtime.
- **FR-010**: Supply utilities to incrementally extend/update the dataset (new identities or labels) while preserving previous manifest versions and enabling re-evaluation.
- **FR-011**: Enforce linting plus unit/integration tests for registry entries, data pipelines, runners, and interfaces; missing required metrics/logs must fail CI.
- **FR-012**: Document installation, dataset prep, configuration usage, interfaces (CLI/Python/REST), evaluation/reproduction steps, and deployment guidance for edge devices.
- **FR-013**: Standardize configs under `configs/<domain>/<name>.yaml`, registry keys as `pkg.component.name`, and `work_dirs/<exp-name>-<timestamp>/` with experiment names following `<task>-<model>-<data>-<id>`.
- **FR-014**: Provide automated LFW download, directory normalization, face detection/alignment, cropping/resizing, checksum generation, and train/val/test split creation with deterministic augmentations via a single command.
- **FR-015**: Surface actionable errors and halt safely on dataset download/checksum/manifest failures without partial outputs; include retry guidance.
- **FR-016**: REST interfaces are internal/no-auth for MVP; production auth is deferred and must be revisited before external exposure.

### Key Entities *(include if feature involves data)*

- **SourceFace**: One or more face crops/embeddings to be injected; includes identity metadata and quality indicators.
- **TargetMedia**: Input image or video frames; includes resolution, frame rate, and detected face locations.
- **ExperimentConfig**: Declarative definition of dataset split, pipeline, model, augmentation, training schedule, evaluation, and export parameters.
- **RunArtifact**: `work_dirs/<exp>/` contents including config snapshot, env hash, logs, metrics (JSON/CSV), visualizations, checkpoints, and sample outputs.
- **MetricReport**: Aggregated metrics and graphs comparing runs and baselines for identity accuracy, quality scores, and latency/FPS.
- **DatasetManifest**: Versioned record of dataset contents, checksums, and splits, including updates/extensions.
- **ExportArtifact**: Optimized model package for deployment (with supported runtime metadata and sample commands).
- **ModelDefinition**: Registry entry describing the encoder-decoder/U-Net architecture, loss components (identity perceptual, adversarial, reconstruction/blending), and compatible exporters.
- **APIContract**: Shared I/O schema covering paths/URLs (plus base64 for REST), batch list handling, controls (face selection, blending strength, frame sampling), and JSON metadata outputs (metrics, artifact URLs).
- **EdgeProfile**: Target deployment profile (e.g., NVIDIA Jetson Orin/Xavier) including memory budget (≤1.5 GB GPU), preferred runtimes (TensorRT/ONNX, FP16), and CPU fallback expectations.
- **ConfigConvention**: Naming/layout standard for `configs/<domain>/<name>.yaml`, registry keys `pkg.component.name`, and `work_dirs/<exp-name>-<timestamp>/` with experiment names `<task>-<model>-<data>-<id>`.

## Assumptions and Dependencies

- Access to the LFW dataset with required licensing notices and checksum manifests.
- Target edge/embedded hardware profiles and performance budgets are provided for benchmarking exports.
- Training and evaluation resources (GPU-capable hardware) are available to meet reproducibility and metric targets.
- Dataset update workflows follow governance for manifest versioning and changelogs.
- Internet access available for automated LFW download during preprocessing; fallback mirrors may be required if primary download fails.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Experiments rerun with the same config/env produce identical metrics (±0.1%) and checkpoints within one hour of elapsed time variance across machines.
- **SC-002**: Identity preservation accuracy on LFW eval split improves by at least 3 percentage points over baseline or meets predefined target; no regression accepted.
- **SC-003**: Quality scores meet thresholds (e.g., LPIPS ≤ target, SSIM/PSNR ≥ targets) while maintaining natural blending; human-rated sample achieves target MOS-style score.
- **SC-004**: Real-time inference achieves ≥30 FPS at 720p on the target edge hardware profile with stable frame-to-frame consistency.
- **SC-005**: Dataset update utilities complete with manifest versioning and changelog, enabling re-evaluation without broken splits or checksum failures.
- **SC-006**: LFW preprocessing yields ≥99% face detection/alignment success on the dataset with 112x112 aligned crops and no checksum/skewed aspect errors across splits.
- **SC-007**: Automated LFW download/prepare command completes without missing files or checksum mismatches and produces validated manifests/splits in under 15 minutes on reference hardware.
