# Research Notes: Face swap system requirements

**Branch**: `001-face-swap-spec`  
**Date**: 2025-11-27  
**Spec**: `/Users/christian/Documents/projects/face_generation/specs/001-face-swap-spec/spec.md`

## Model & Loss Design

- **Decision**: Encoder-decoder/U-Net generator with identity perceptual loss (ArcFace features), adversarial loss (multi-scale discriminator), and reconstruction/blending losses; temporal smoothing hook for video inference.
- **Rationale**: Balances fidelity and efficiency; integrates cleanly with registry-based components and edge export to ONNX/TensorRT; perceptual + adversarial maintain identity/expression/skin tone.
- **Alternatives considered**: Diffusion-based swapper (higher quality, heavier, slower); GAN with style modulation only (risk of instability, harder to export/lightweight).

## Data Pipeline (LFW)

- **Decision**: RetinaFace detection, 5-point alignment to 112x112 crops, LFW split 80/10/10, light augmentations (color jitter, mild blur, horizontal flip), checksum-validated manifests and versioned dataset updates; automated Kaggle download and folder normalization via preprocessing script with validation.
- **Rationale**: Maximizes identity fidelity while adding minimal variance; aligns with FR and constitution data integrity requirements; automation improves reproducibility.
- **Alternatives considered**: MTCNN only (less accurate); heavy augmentations (risk identity loss); manual download (less reproducible).

## Metrics & Evaluation

- **Decision**: Identity accuracy via ArcFace embeddings; LPIPS/SSIM/PSNR for quality; latency/FPS for video; MOS-style small human rating sample; trend/comparison graphs; regression check vs baseline; example thresholds: identity accuracy no regression, LPIPS ↓, SSIM/PSNR ↑ vs baseline, FPS ≥ 30 at 720p edge target.
- **Rationale**: Covers identity, perceptual quality, and real-time readiness; human sample adds qualitative guardrail.
- **Alternatives considered**: Automated metrics only (misses visual artifacts); larger human panels (higher cost).

## Interfaces & I/O

- **Decision**: Shared schema across CLI/Python/REST: inputs accept file/URL paths (REST also base64), batch lists; controls for face selection, blending strength, frame sampling; outputs include artifacts and JSON metadata (metrics, URLs).
- **Rationale**: Consistency across interfaces reduces maintenance and enforces single pipeline.
- **Alternatives considered**: Single-item only (limits batch use cases); base64-only (hurts large video flows).

## Edge/Embedded Deployment

- **Decision**: Export ONNX → TensorRT (FP16 preferred) for Jetson-class (Orin/Xavier) with ≤1.5 GB GPU mem; CPU fallback via ONNX Runtime; include edge benchmark command and profiling hooks; target 720p @ ≥30 FPS.
- **Rationale**: Meets target hardware and latency; uses widely supported runtimes.
- **Alternatives considered**: NNAPI/Android NPU (out of scope for Jetson target); x86-only export (misses edge goal).

## Configuration & Naming

- **Decision**: Configs under `configs/<domain>/<name>.yaml`; registry keys `pkg.component.name`; `work_dirs/<exp-name>-<timestamp>/`; experiment names `<task>-<model>-<data>-<id>`.
- **Rationale**: Enforces reproducibility and discoverability across contributors.
- **Alternatives considered**: Free-form naming (hurts reproducibility); flat config directory (harder to navigate).

## Testing & Quality Gates

- **Decision**: pytest unit/integration for registry components, data pipelines, runner, interfaces; contract tests for REST/CLI I/O; lint via ruff/mypy; CI block on missing metrics/logs; reference config reproduction check.
- **Rationale**: Aligns with constitution quality gates and real-time reliability.
- **Alternatives considered**: Lint-only (insufficient), manual QA (not reproducible).
