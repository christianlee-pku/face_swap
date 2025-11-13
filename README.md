# FaceSwap Lite — Research & Production Template

> **Lightweight face swap pipeline** that runs on laptops and edge devices. Trains on **LFWPeople**, swaps faces on **images** and **videos**, offers a simple **HTTP API**, and includes **evaluation graphs** and **dataset update** tooling. All components are written with deployment in mind (clean dependencies, small models).

<p align="center">
  <img alt="faceswap-lite" src="https://img.shields.io/badge/face--swap-lite-blueviolet?logo=pytorch">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%7C3.11-informational">
  <img alt="torch" src="https://img.shields.io/badge/torch-2.2.2-red">
  <img alt="fastapi" src="https://img.shields.io/badge/FastAPI-ready-brightgreen">
</p>

---

## Task

- **Dataset**: [Kaggle · LFWPeople](https://www.kaggle.com/datasets/atulanandjha/lfwpeople) → `data/raw/lfw/<IDENTITY>/*.jpg`  
- **Preprocess** (detect & align to 256x256) → `data/aligned_256/` + pairs CSV  
- **Train** two stages on mobile-friendly nets (**Stage A**: ID encoder + warmup generator; **Stage B**: freeze ID encoder, train generator)  
- **Evaluate** identity cosine, PSNR, and draw accuracy graphs  
- **Infer** on images/videos; **real‑time** webcam optional  
- **Serve** via FastAPI: `/swap` 
- **Update data** with `scripts/update_dataset.py` (incremental ingest & re-sample pairs)

---

## Features

- **Mobile‑friendly architectures** (`MobileIDEnc`, `MobileSwapLite`) ready for ONNX/TensorRT export (not included by default)
- **5-point ArcFace alignment** (MTCNN detect → similarity warp to 256×256 template)
- **Evaluation suite**: identity cosine (cos(out,src), cos(out,tgt)), margin hist, PSNR hist, scatter plots
- **FastAPI service** for production integration
- **Dataset updater** to ingest new identities and rebuild pairs
- **Video swapping** with temporal smoothing (EMA + mask blending) and adjustable detection cadence
- **Clear logs** & simple structure (no deep nesting)

---

## Project Structure

```
face_swap/
├─ scripts/
│  ├─ preprocess_lfw.py        # Phase 1: detect+align(256) + images index + pairs csv
│  ├─ update_dataset.py        # Incremental ingest: new_raw -> aligned -> pairs
│  ├─ train.py                 # Phase 2: Stage A/B training with clear logging
│  ├─ eval_accuracy.py         # Phase 3: metrics + accuracy graphs
│  ├─ infer_image.py           # Phase 4: image swapping CLI
│  ├─ infer_video.py           # Phase 5: video swapping w/ temporal smoothing
│  └─ rt_face_swap.py          # (optional) real-time webcam / stream
│
├─ service/
│  └─ app.py                   # FastAPI app: /swap
│
├─ utils/
│  ├─ face_align.py            # MTCNN 5 landmarks + 256 warp (ArcFace template)
│  ├─ infer_core.py            # Inference engine + blending/mask
│  ├─ train_utils.py           # Models (MobileIDEnc/MobileSwapLite) + loops
│  ├─ losses.py                # MobileNetV2 perceptual + L1
│  ├─ image_io.py              # PIL/NumPy helpers
│  └─ logging_utils.py         # file+console logger
│
├─ data/
│  ├─ raw/lfw/                 # LFW extracted here (folder per identity)
│  ├─ lfw_aligned/             # Aligned 256×256 faces (auto-created)
│  ├─ assets/                  # Demo images/video (src.png, tgt.png, demo.mp4)
│  └─ metadata/                # images.csv, pairs_train.csv, pairs_val.csv
│
├─ work_dirs/
│  ├─ train/ckpts/             # id_encoder_fixed.pt, gen_A_last.pt, gen_B_best.pt
│  ├─ train/logs/              # training logs
│  └─ eval/                    # eval csv/json + graphs
│
├─ requirements.txt
├─ pipeline.sh                 # End-to-end pipeline orchestrator
└─ README.md
```

---

## Install

> **Python** 3.10–3.11 recommended. We deliberately pin NumPy 1.26.x to avoid ABI conflicts with older wheels.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data

1. Download **LFWPeople** from Kaggle and extract into `data/lfw_raw/` such that files look like:
   ```
   data/lfw_raw/George_W_Bush/img_0001.jpg
   data/lfw_raw/George_W_Bush/img_0002.jpg
   ...
   ```
2. (Optional) Put demo test files into `data/assets/`: `src.png`, `tgt.png`, and `demo.mp4`

---

## One‑Command Pipeline

```bash
chmod +x pipeline.sh
./pipeline.sh
```

What it does: preprocess → (optional) update dataset → Stage A → Stage B → evaluate & graphs → image inference → start HTTP service → optional video inference.

---

## Training

### Stage A (ID warmup + generator)
```bash
python -m scripts.train   --stage A   --pairs-csv data/metadata/pairs_train.csv   --work-dir ./work_dirs/train/ckpts/   --device cpu   --epochs 5   --batch-size 8   --lr 2e-4   --w-perc 0.3   --w-l1-face 0.4   --w-l1-bg 1.0   --w-id 1.0   --stageA-same-only true
```
- Saves `work_dirs/train/ckpts/id_encoder_fixed.pt` and `gen_A_last.pt`

### Stage B (Freeze ID encoder, train generator)
```bash
python -m scripts.train   --stage B   --pairs-csv data/metadata/pairs_train.csv   --val-csv   data/metadata/pairs_val.csv   --work-dir work_dirs/train   --device cpu   --epochs 5   --batch-size 8   --lr 2e-4   --w-id 2.0   --w-perc 0.3   --w-l1-face 0.2   --w-l1-bg 1.0   --id-encoder-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt   --init-gen work_dirs/train/ckpts/gen_A_best.pt
```
- Saves `work_dirs/train/ckpts/gen_B_best.pt`

> **Tip:** If outputs look like the target (no identity change), increase Stage‑B epochs and diversify training pairs (ensure many **different‑identity** pairs).

---

## Evaluation & Accuracy Graphs

```bash
python -m scripts.eval_accuracy   --val-csv data/metadata/pairs_val.csv   --ckpt work_dirs/train/ckpts/gen_B_best.pt   --id-enc-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt   --device cpu   --out-dir work_dirs/eval   --max-samples 800
```
Artifacts in `work_dirs/eval/`:
- `idcos_hist.png` — histogram of cos(out, src)
- `idcos_tgt_hist.png` — histogram of cos(out, tgt)
- `id_margin_hist.png` — histogram of cos(out,src) − cos(out,tgt)
- `id_scatter.png` — scatter of cos(out,src) vs cos(out,tgt)
- `psnr_hist.png` — PSNR(out, tgt) distribution  
- `eval_summary.json` — summary stats

---

## Inference

### Image
```bash
python -m scripts.infer_image   --src data/assets/src.png   --tgt data/assets/tgt.png   --ckpt work_dirs/train/ckpts/gen_B_best.pt   --id-enc-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt   --out data/assets/out.png   --blend feather   --feather-ksize 41   --color-match     --mask-scale-x 0.95   --mask-scale-y 0.90
```

### Video (temporal smoothing)
```bash
python -m scripts.infer_video   --src data/assets/src.png   --tgt-video data/assets/demo.mp4   --out-video work_dirs/infer/demo_swap.mp4   --ckpt work_dirs/train/ckpts/gen_B_best.pt   --id-enc-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt   --device cpu   --blend feather --feather-ksize 41   --color-match   --detect-every 3   --smooth-alpha 0.6   --temporal-alpha 0.75   --mask-scale-x 0.95 --mask-scale-y 0.9
```

---

## Service (FastAPI)

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
- `POST /swap` — multipart form:
  - `src` (file), `tgt` (file), optional: `blend`, `feather_ksize`, `color_match` (bool), `token`

Example:
```bash
curl -X POST http://127.0.0.1:8000/swap   -F "src=@data/assets/src.png"   -F "tgt=@data/assets/tgt.png"   -F "blend=feather"   -F "feather_ksize=31"   -F "color_match=true"   -o out.png
```

---

## Performance Tips & Real‑Time Notes

- **Detection cadence**: For videos, detect every *N* frames (`--detect-every`) and smooth landmarks (EMA).
- **Temporal blending**: Use `--temporal-alpha` to mix current & previous outputs within the face mask to avoid flicker.
- **Batching**: For image batches, pre‑align multiple targets and run the generator in mini‑batches.
- **Mixed precision**: Enable AMP on GPU to reduce latency (not included by default in this CPU‑friendly template).
- **Precompute masks**: `scripts/precompute_masks.py` generates soft face masks to save per‑frame compute.
- **Mobile/Edge**: Models are small and use standard ops → suitable for ONNX export; consider post‑training **quantization**.

---

## Dataset Update (Incremental)

Add new identities under `data/new_raw/<IDENTITY>/*.jpg`, then run:
```bash
python -m scripts.update_dataset   --new-raw-dir data/new_raw   --dst-align-dir data/lfw_aligned   --index-csv data/metadata/images.csv   --pairs-train-csv data/metadata/pairs_train.csv   --pairs-val-csv data/metadata/pairs_val.csv   --min-per-id 2 --same-ratio 0.2 --val-ratio 0.1 --max-pairs 20000
```

---

## Troubleshooting

- **Pydantic v2**: We already use `pydantic-settings` for `BaseSettings`. Do **not** import `BaseSettings` from `pydantic`.
- **NumPy 1.x vs 2.x ABI**: We pin `numpy==1.26.4` to avoid wheel incompatibilities. If you must use 2.x, rebuild native deps.
- **MTCNN CPU speed**: It’s okay for LFW and demos. For production, consider a lighter 5‑pt detector or cache detections.
- **Outputs look like target (no swap)**: Increase Stage‑B epochs, ensure enough **different‑identity** pairs, and verify that `id_encoder_fixed.pt` is correctly loaded/frozen in Stage‑B.

---

## Roadmap

- [ ] Export ONNX + TensorRT samples (desktop & mobile)
- [ ] INT8/FP16 quantization recipes
- [ ] Replace MTCNN with a lighter/faster 5pt detector
- [ ] Add Poisson blending option for edges
- [ ] Web UI demo (Gradio/Streamlit)

---

## Ethics & Legal

This repository is for **research, education, and authorized use only**.  
You are responsible for complying with laws, policies, and obtaining **clear consent** from individuals whose likeness is used. Misuse may violate privacy, publicity, or other rights.

---

## License

MIT (add your name and year). Third‑party models/libraries are subject to their own licenses.

---

## Acknowledgements

- **LFW** dataset authors & Kaggle host
- **facenet-pytorch** for MTCNN
- **PyTorch / TorchVision** for training & MobileNetV2 features
- Community research on ArcFace 5‑pt alignment

---
