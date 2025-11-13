# FaceSwap Lite — Task Handbook

*A practical, full‑English guide to every task referenced in the README — what each script does, required inputs/outputs, recommended commands, and how to debug and tune for performance.*

> **Scope**: `scripts/preprocess_lfw.py`, `scripts/update_dataset.py`, `scripts/train.py`, `scripts/eval_accuracy.py`, `scripts/infer_image.py`, `scripts/infer_video.py`, `service/app.py`, `scripts/precompute_masks.py`, `scripts/rt_face_swap.py` (architecture), and `pipeline.sh` orchestration.

---

## Table of Contents

1. [Conventions & Paths](#conventions--paths)  
2. [Phase 1 — `preprocess_lfw.py`](#phase-1--preprocess-lfwpy)  
3. [Dataset Updates — `update_dataset.py`](#dataset-updates--update_datasetpy)  
4. [Phase 2 — Training (`train.py`)](#phase-2--training-trainpy)  
5. [Phase 3 — Evaluation & Accuracy Graphs (`eval_accuracy.py`)](#phase-3--evaluation--accuracy-graphs-eval_accuracypy)  
6. [Phase 4 — Image Inference (`infer_image.py`)](#phase-4--image-inference-infer_imagepy)  
7. [Phase 5 — Video Inference (`infer_video.py`)](#phase-5--video-inference-infer_videopy)  
8. [Service — HTTP API (`service/app.py`)](#service--http-api-serviceapppy)  
9. [Optional — Precompute Masks (`precompute_masks.py`)](#optional--precompute-masks-precompute_maskspy)  
10. [Optional — Real‑Time Architecture (`rt_face_swap.py`)](#optional--realtime-architecture-rt_face-swappy)  
11. [Pipeline Orchestration (`pipeline.sh`)](#pipeline-orchestration-pipelinesh)  
12. [Reproducibility & Versioning](#reproducibility--versioning)  
13. [Performance & Deployment Notes](#performance--deployment-notes)  
14. [FAQ & Troubleshooting](#faq--troubleshooting)

---

## Conventions & Paths

- **Dataset**: [Kaggle · LFWPeople](https://www.kaggle.com/datasets/atulanandjha/lfwpeople)  
  Extract as `data/raw/lfw/<IDENTITY>/*.jpg` (folder per identity).  
- **Aligned faces**: `data/lfw_aligned/` (auto‑created at 256×256).  
- **Metadata CSVs**: `data/metadata/image_index.csv`, `pairs_train.csv`, `pairs_val.csv`.  
- **Checkpoints**: `work_dirs/train/ckpts/` → `id_encoder_fixed.pt`, `gen_A_last.pt`, `gen_B_best.pt`.  
- **Logs/Graphs**: `work_dirs/train/logs/`, `work_dirs/eval/`.  
- **Demo assets**: `data/assets/src.png`, `data/assets/tgt.png`, `data/assets/demo.mp4` (optional).

> **Tip** — Keep identities balanced. A minimum of **2 images per identity** is required to form “same” pairs.

---

## Phase 1 — `preprocess_lfw.py`

### What it does (Purpose)
1. **Detects** 5 facial landmarks per image (MTCNN).  
2. **Aligns** faces to the ArcFace 5‑point template at **256×256** via similarity transform.  
3. **Indexes** aligned images into `image_index.csv` with `identity`, `path`, `md5`, `width`, `height`.  
4. **Samples** training/validation pairs (`pairs_train.csv`, `pairs_val.csv`) with a controllable **same:diff** ratio.

### Inputs
- `--lfw-raw`: Root of raw LFW folder structure (`data/raw/lfw`).

### Outputs
- `--align-out`: `data/aligned_256/` (aligned PNGs).  
- `--images-index`: `data/metadata/image_index.csv`.  
- `--pairs-train`, `--pairs-val`: sampled pair CSVs.

### Example
```bash
python -m scripts.preprocess_lfw
```

### Verification Checklist
- Aligned faces exist under `data/aligned_256/<IDENTITY>/*.png`.  
- `image_index.csv` row count ≈ aligned images count.  
- `pairs_train.csv` and `pairs_val.csv` contain expected columns: `src,tgt,same`.

---

## Dataset Updates — `update_dataset.py`

### Purpose
Incrementally **ingest new raw images**, **align** them the same way, **rebuild** `image_index.csv`, and **re‑sample** pairs.

### Inputs
- `--new-raw-dir`: `data/new_raw/<IDENTITY>/*.jpg` for newly collected data.  
- `--dst-align-dir`: where aligned outputs are stored (`data/lfw_aligned`).

### Outputs
- Updated `data/metadata/image_index.csv`.  
- Fresh `pairs_train.csv`, `pairs_val.csv` (keeps your training split current).

### Example
```bash
python -m scripts.update_dataset   --new-raw-dir data/new_raw   --dst-align-dir data/aligned_256   --index-csv data/metadata/image_index.csv   --pairs-train-csv data/metadata/pairs_train.csv   --pairs-val-csv data/metadata/pairs_val.csv   --min-per-id 2 --same-ratio 0.2 --val-ratio 0.1 --max-pairs 20000
```

### Notes
- Uses **exactly the same alignment** as preprocessing for consistency.  
- Rebuilds the **entire index** to avoid drift and duplication.  
- Consider versioning the CSVs if you want to compare old vs new training runs.

---

## Phase 2 — Training (`train.py`)

### Overview
Two stages to stabilise identity control and visual quality with **mobile‑friendly** models.

- **Stage A**: Train a light **ID encoder** (`MobileIDEnc`) and a warm‑up generator (`MobileSwapLite`). Often use `--stageA-same-only true` so generator first learns a conservative reconstruction task.  
- **Stage B**: **Freeze** the ID encoder; train the generator to **map target content + source identity**.

### Models & Loss
- **ID Encoder**: small ConvNet, global pooling, L2‑normalised embedding (default 256‑d).  
- **Generator**: encoder‑decoder taking target image + projected ID embedding; outputs swapped face (sigmoid to [0,1]).  
- **Loss**: **Perceptual (MobileNetV2 features) + L1** for stable gradients and mobile‑friendly ops.

### Inputs
- `--pairs-csv`: usually `data/metadata/pairs_train.csv` from Phase 1.  
- Optional warm start for Stage B: `--id-encoder-ckpt`, `--init-gen` (from Stage A).

### Outputs
- **Checkpoints**: `work_dirs/train/ckpts/`  
  - Stage A: `id_encoder_fixed.pt`, `gen_A_last.pt`  
  - Stage B: `gen_B_best.pt` (saved per epoch in this template)  
- **Logs**: `work_dirs/train/logs/train_*.log`

### Key Arguments
- `--stage A|B` — which phase to run.  
- `--epochs`, `--batch-size`, `--lr`, `--save-every` — standard training knobs.  
- `--stageA-same-only true|false` — Stage A target = **tgt** (true) or **src** (false).  
- `--id-encoder-ckpt`, `--init-gen` — Stage B initialisation.

### Commands
**Stage A**
```bash
python -m scripts.train   --stage A   --pairs-csv data/metadata/pairs_train.csv   --work-dir ./work_dirs/train/ckpts/   --device cpu   --epochs 5   --batch-size 8   --lr 2e-4   --w-perc 0.3   --w-l1-face 0.4   --w-l1-bg 1.0   --w-id 1.0   --stageA-same-only true
```
**Stage B**
```bash
python -m scripts.train   --stage B   --pairs-csv data/metadata/pairs_train.csv   --val-csv   data/metadata/pairs_val.csv   --work-dir work_dirs/train   --device cpu   --epochs 5   --batch-size 8   --lr 2e-4   --w-id 2.0   --w-perc 0.3   --w-l1-face 0.2   --w-l1-bg 1.0   --id-encoder-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt   --init-gen work_dirs/train/ckpts/gen_A_best.pt
```

### Debugging & Tips
- **Output ≈ target (identity not changing)**:  
  - Increase **Stage B epochs** and ensure plenty of **different‑identity** pairs.  
  - Confirm **ID encoder is frozen** during Stage B.  
  - Check learning rate; too low may underfit, too high may destabilise.  
- **Loss plateaus early**: expand dataset, raise `--max-pairs`, or reduce `--stageA-same-only` reliance.  
- **CPU training** is slow; prefer GPU if available.

---

## Phase 3 — Evaluation & Accuracy Graphs (`eval_accuracy.py`)

### Purpose
Quantify how well the model transfers the source identity while preserving target content.

### Metrics
- **cos(out, src)** — higher is **better** (identity matches source).  
- **cos(out, tgt)** — lower is **better** (output identity diverges from target identity).  
- **Margin**: cos(out,src) − cos(out,tgt) — higher margin indicates stronger identity transfer.  
- **PSNR(out, tgt)** — measures reconstruction fidelity to target background/lighting.

### Outputs
- Graphs: `idcos_hist.png`, `idcos_tgt_hist.png`, `id_margin_hist.png`, `id_scatter.png`, `psnr_hist.png`  
- JSON: `eval_summary.json` with aggregate stats.

### Example
```bash
python -m scripts.eval_accuracy   --val-csv data/metadata/pairs_val.csv   --ckpt work_dirs/train/ckpts/gen_B_best.pt   --id-enc-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt   --device cpu   --out-dir work_dirs/eval   --max-samples 800
```

### Interpreting Results
- Expect **cos(out,src)** to shift right (toward 1.0) as training improves.  
- **Margin** histogram should be positive‑skewed.  
- **PSNR** is complementary — too high may imply the model is copying target; balance identity vs content.

---

## Phase 4 — Image Inference (`infer_image.py`)

### What it does
- Aligns **src** & **tgt**, encodes **src identity**, runs generator, warps output back, and **blends** onto target.  
- Optional **color‑matching** to reduce tone mismatch.  
- Soft **ellipse mask** with Gaussian feathering for natural seams.

### Arguments
- `--src`, `--tgt`, `--ckpt`, `--id-enc-ckpt`, `--out`, `--device`  
- `--blend feather|poisson` (template implements feather)  
- `--feather-ksize` (odd int), `--color-match` (flag)

### Example
```bash
python -m scripts.infer_image   --src data/assets/src.png   --tgt data/assets/tgt.png   --ckpt work_dirs/train/ckpts/gen_B_best.pt   --id-enc-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt   --out data/assets/out.png   --blend feather   --feather-ksize 41   --color-match     --mask-scale-x 0.95   --mask-scale-y 0.90
```

### Notes
- If **no faces detected**, the script returns target unchanged (by design).  
- Increasing `--feather-ksize` softens seams but may blur details.

---

## Phase 5 — Video Inference (`infer_video.py`)

### Pipeline
1. Read frames, convert to RGB.  
2. For every frame, perform image swap.  
3. **Detect landmarks** periodically (`--detect-every N`) and apply **EMA** smoothing to landmarks.  
4. Construct soft **face mask** and apply **temporal blending** to suppress flicker.  
5. Write frames to output video.

### Key Arguments
- `--detect-every` (default 3): run landmark detection every N frames to save compute.  
- `--smooth-alpha` (default 0.6): EMA smoothing for landmarks (0→no memory, 1→very slow updates).  
- `--temporal-alpha` (default 0.75): mix current/previous faces within mask.  
- `--mask-scale-x`, `--mask-scale-y`: ellipse mask extents.  
- `--color-match`: stabilises tone across cuts.

### Example
```bash
python -m scripts.infer_video   --src data/assets/src.png   --tgt-video data/assets/demo.mp4   --out-video work_dirs/infer/demo_swap.mp4   --ckpt work_dirs/train/ckpts/gen_B_best.pt   --id-enc-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt   --device cpu   --blend feather --feather-ksize 41   --color-match   --detect-every 3   --smooth-alpha 0.6   --temporal-alpha 0.75   --mask-scale-x 0.95 --mask-scale-y 0.9
```

### Performance Hints
- Lower `--detect-every` (e.g., 5–10) for static shots; higher for fast motion.  
- Set `--temporal-alpha` near **0.8–0.9** for stronger temporal coherence.  
- Consider precomputing masks (see next section) to reduce per‑frame cost.

---

## Service — HTTP API (`service/app.py`)

### Purpose
Expose a **production‑ready** image‑to‑image swap API.

### Environment Variables
- `GEN_CKPT`, `ID_ENC_CKPT`, `DEVICE`, `MAX_IMAGE_MB`, `OUTPUT_DIR`, `TOKEN` (optional).

### Start
```bash
export GEN_CKPT=work_dirs/train/ckpts/gen_B_best.pt
export ID_ENC_CKPT=work_dirs/train/ckpts/id_encoder_fixed.pt
export DEVICE=cpu
export MAX_IMAGE_MB=10
export OUTPUT_DIR=work_dirs/services

uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 1
```

### Endpoints
- `POST /swap` → multipart form:
  - **Fields**: `src` (file), `tgt` (file), `blend`, `feather_ksize`, `color_match` (bool), `token` (optional)  
  - **Response**: PNG stream (swapped image)

### Example Client Call
```bash
curl -X POST http://127.0.0.1:8000/swap   -F "src=@data/assets/src.png"   -F "tgt=@data/assets/tgt.png"   -F "blend=feather" -F "feather_ksize=31" -F "color_match=true"   -o out.png
```

### Notes
- If you set `TOKEN`, you **must** include it as a form field; otherwise requests will be rejected (401).  
- For production, front with a gateway (rate‑limit, authN/Z, logging).

---

## Optional — Precompute Masks (`precompute_masks.py`)

### Purpose
Pre‑generate **soft ellipse masks** for aligned images to speed up inference or analysis tasks.

### Example
```bash
python -m scripts.precompute_masks   --aligned-dir data/lfw_aligned   --out-dir data/mask_256   --scale-x 1.0 --scale-y 1.0 --feather-ksize 31
```
Outputs `data/mask_256/<IDENTITY>/*.png` masks aligned to each 256×256 face.

---

## Optional — Real‑Time Architecture (`rt_face_swap.py`)

### Design (high‑level)
- **Capture Thread** → pulls frames from webcam/RTSP, places in bounded queue.  
- **Worker Thread** → performs swap; uses **drop policy** if queue grows (skip frames rather than lag).  
- **Display/Output Thread** → shows frames, handles UI/recording.  
- **EMA landmarks** + **temporal blending** for stability; run detection every *N* frames.

> This template leaves real‑time code as an exercise; follow the video pipeline, add threads/queues, and choose a drop strategy (e.g., “latest wins”).

---

## Pipeline Orchestration (`pipeline.sh`)

### What it does
1. **Preprocess** LFW → aligned faces + pairs.  
2. **Update dataset** if `data/new_raw/` exists.  
3. Train **Stage A**, then **Stage B**.  
4. **Evaluate** & save graphs.  
5. **Infer (image)** with demo assets.  
6. **Start service** (Uvicorn).  
7. Optionally **infer (video)** if `data/assets/demo.mp4` is present.

### How to run
```bash
chmod +x pipeline.sh
./pipeline.sh
```

### Where to look
- Training logs: `work_dirs/train/logs/`  
- Checkpoints: `work_dirs/train/ckpts/`  
- Eval results: `work_dirs/eval/`  
- Service logs: `uvicorn.log` (if you start via `nohup` inside the script)

---

## Reproducibility & Versioning

- **Python** 3.10–3.11; requirements pinned (notably `numpy==1.26.4`).  
- Seeded sampling (`--seed`) for deterministic pair generation.  
- Version your `data/metadata/*.csv` and `work_dirs/train/ckpts/*` when comparing experiments.  
- Keep a copy of `requirements.txt` alongside your checkpoints.

---

## Performance & Deployment Notes

- Prefer **GPU** during training; the network is small but identity learning benefits from more data/epochs.  
- For **mobile/edge** deployment, export to **ONNX** then run **TensorRT** or on‑device runtimes (NNAPI, CoreML), plus **post‑training quantisation**.  
- Replace **MTCNN** with a lighter 5‑pt detector for production latency.  
- Batch **multiple target faces** during inference to improve throughput on GPU servers.

---

## FAQ & Troubleshooting

**Q: My output looks almost the same as the target (no identity swap).**  
A: Train **Stage B** longer; add **different‑identity** pairs; verify **ID encoder is frozen**; consider higher LR or stronger perceptual weight.

**Q: Preprocess skipped many images.**  
A: Some LFW images are profile/occluded. Try a more robust detector or adjust thresholds; ensure `min-per-id` doesn’t filter too hard.

**Q: Service returns 401.**  
A: You set `TOKEN` in the environment; include it in the request form as `token=<YOUR_TOKEN>`.

**Q: NumPy 1.x vs 2.x error.**  
A: Use `numpy==1.26.4` (as in the template) or rebuild native dependencies for 2.x.

**Q: Video flickers.**  
A: Increase `--temporal-alpha`, reduce detection frequency (`--detect-every`), and enable `--color-match` for tone stability.

---

**You’re all set.** Use this handbook as the single source of truth for what every task does, why it exists, and how to run it confidently.
