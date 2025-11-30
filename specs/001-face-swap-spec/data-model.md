# Data Model: Face swap system requirements

**Branch**: `001-face-swap-spec`  
**Date**: 2025-11-27  
**Spec**: `/Users/christian/Documents/projects/face_generation/specs/001-face-swap-spec/spec.md`

## Entities

- **SourceFace**
  - Fields: image path/URL/base64, face bbox/landmarks, identity embedding, quality score, metadata (id, license note).
  - Rules: face detection required; low-quality flagged; optional multiple sources per request.

- **TargetMedia**
  - Fields: media path/URL/base64, media type (image/video), resolution, frame rate, detected face bboxes/landmarks per frame, normalization params.
  - Rules: alignment required; supports batch lists; temporal ordering preserved for video.

- **ExperimentConfig**
  - Fields: dataset selection/splits, model registry key + hyperparameters, augmentations, training schedule, evaluation suite, export settings, seed.
  - Rules: lives in `configs/<domain>/<name>.yaml`; must be self-contained; no hidden defaults.

- **RunArtifact**
  - Fields: config snapshot, env hash, logs (structured), metrics (JSON/CSV), visualizations/graphs, checkpoints (best + last K), sample outputs, reproduction README.
  - Rules: stored in `work_dirs/<exp-name>-<timestamp>/`; immutable snapshots.

- **MetricReport**
  - Fields: identity accuracy, LPIPS/SSIM/PSNR, latency/FPS, human rating summary, baseline comparison deltas.
  - Rules: must include regression check vs baseline; serialized to JSON/CSV + graphs.

- **DatasetManifest**
  - Fields: version, checksums, split definitions, preprocessing/alignment parameters, changelog for updates/extensions.
  - Rules: version bumps required on changes; checksum validation mandatory.

- **ExportArtifact**
  - Fields: ONNX/TensorRT package, supported precision (FP16/FP32), runtime metadata, sample commands, edge benchmark results.
  - Rules: targets Jetson-class (≤1.5 GB GPU mem); must include CPU fallback notes (ONNX Runtime).

- **ModelDefinition**
  - Fields: architecture (encoder-decoder/U-Net), loss components (identity perceptual, adversarial, reconstruction/blending), registry key, default config references.
  - Rules: registered with docs/tests; compatible with exporters.

- **APIContract**
  - Fields: input schema (paths/URLs + base64 for REST), batch list support, controls (face selection, blending strength, frame sampling), output schema (artifacts + JSON metadata, metric URLs).
  - Rules: shared across CLI/Python/REST; validation on required fields.

- **EdgeProfile**
  - Fields: device class (Jetson Orin/Xavier), memory budget (≤1.5 GB GPU), target FPS/resolution (≥30 FPS @ 720p), preferred runtimes (TensorRT/ONNX), fallback (ONNX Runtime CPU).
  - Rules: benchmarks logged in `work_dirs` artifacts; failing targets must flag.

- **ConfigConvention**
  - Fields: naming pattern `<task>-<model>-<data>-<id>`, registry key format `pkg.component.name`, directory layout `configs/<domain>/<name>.yaml`, `work_dirs/<exp-name>-<timestamp>/`.
  - Rules: required for all new configs/experiments.
