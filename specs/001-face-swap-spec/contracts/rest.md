# REST Contract: Face swap system requirements

**Branch**: `001-face-swap-spec`  
**Date**: 2025-11-27  
**Spec**: `/Users/christian/Documents/projects/face_generation/specs/001-face-swap-spec/spec.md`

## Base

- Base URL: `/api/v1`
- Content types: `application/json` for metadata; `multipart/form-data` for file/base64 payloads when needed.
- Auth: Internal/no-auth for MVP; production auth must be added before external exposure.

## Endpoints

### POST `/face-swap/batch`
- Purpose: Batch image/video swapping.
- Request:
  - `sources`: list of {path|url|base64}
  - `targets`: list of {path|url|base64}
  - `blend_strength` (optional, float 0-1)
  - `frame_sampling` (optional, e.g., `all`, `stride:<n>`)
  - `face_selection` (optional, index or strategy)
  - `config` (optional) path/ref to config in `configs/`
- Response:
  - `artifacts`: list of result URLs/paths
  - `metrics`: identity accuracy (if eval), LPIPS/SSIM/PSNR (if eval), latency/FPS (if video)
  - `work_dir`: path to run artifacts
  - `errors`: list of per-item errors if any

### POST `/face-swap/stream`
- Purpose: Real-time/near-real-time video swapping (streaming request/response chunking).
- Request:
  - `source`: {path|url|base64}
  - `target_stream`: {url|stream id}
  - `frame_sampling` (optional)
  - `config` (optional)
- Response:
  - `stream_url` for consumed output or `ws` endpoint
  - `metrics`: latency/FPS
  - `work_dir`

### GET `/reports/{run_id}`
- Purpose: Retrieve metrics/graphs for a run.
- Response:
  - `metrics`: JSON/CSV links
  - `graphs`: URLs to plotted visuals
  - `samples`: URLs to sample frames/clips

## Error Handling
- 400 for validation errors (missing sources/targets).
- 422 for invalid media (no face detected).
- 500 for internal errors with run_id for log correlation.
