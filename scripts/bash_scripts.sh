
# =========================
# 0) Preprocess LFW
#    - detect & align faces to 256x256
#    - build images index & train/val pairs
# =========================
python -m scripts.preprocess_lfw

# =========================
# 2) Train Stage A (identity encoder)
#    - same-identity emphasis (reconstruction-like)
#    - outputs fixed ID encoder ckpt
#    Train Stage B (generator with fixed id encoder)
#    - cross-identity swap training
#    - init generator from Stage A last/best if applicable
# =========================
python -m scripts.train \
  --train-csv data/metadata/pairs_val.csv \
  --val-csv   data/metadata/sample_pairs_train.csv \
  --stageA-epochs 3 --stageA-same-only \
  --stageB-epochs 2 \
  --batch-size 4 --lr 2e-4 \
  --w-perc 0.3 --w-l1-face 0.4 --w-l1-bg 1.0 --w-id 1.0
  --log-interval 10 --log-grad-norm \
  --out-dir ./work_dirs/train/ckpts/

python -m scripts.train \
  --train-csv data/metadata/sample_pairs_val.csv \
  --val-csv   data/metadata/sample_pairs_train.csv \
  --stageA-epochs 0 \
  --stageB-epochs 2 \
  --batch-size 8 --lr 2e-4 \
  --w-id 2.0 --w-perc 0.3 --w-l1-face 0.2 --w-l1-bg 1.0 \
  --num-workers 4 \
  --out-dir ./work_dirs/train/ckpts \
  --log-interval 10 --log-grad-norm \
  --init-gen ./work_dirs/train/ckpts/gen_A_best.pt \
  --id-enc-ckpt ./work_dirs/train/ckpts/id_encoder_fixed.pt

# =========================
# 3) Evaluation & accuracy graphs
#    - identity cosine, margin hist, PSNR, ROC/AUC
# =========================
# evaluate a single checkpoint
# python -m scripts.eval \
#   --val-csv data/metadata/sample_pairs_train.csv \
#   --ckpt ./work_dirs/train/ckpts/gen_B_last.pt \
#   --batch-size 8 \
#   --out-dir ./work_dirs/eval

# evaluate all B-stage checkpoints and draw curves
python -m scripts.eval \
  --val-csv data/metadata/sample_pairs_train.csv \
  --ckpt-glob "work_dirs/train/ckpts/gen_B_*.pt" \
  --batch-size 8 \
  --out-dir ./work_dirs/eval/

python -m scripts.eval_accuracy \
  --val-csv data/metadata/sample_pairs_train.csv \
  --ckpt work_dirs/train/ckpts/gen_B_best.pt \
  --id-enc-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt \
  --device cpu \
  --out-dir work_dirs/eval \
  --max-samples 800

# =========================
# 4) Image inference with trained-well model
# =========================
python -m scripts.infer_image \
  --src ./data/assets/src.png \
  --tgt ./data/assets/tgt.png \
  --ckpt ./work_dirs/train/ckpts/gen_B_last.pt \
  --id-enc-ckpt ./work_dirs/train/ckpts/id_encoder_fixed.pt \
  --out ./data/assets/out.png \
  --blend feather --feather-ksize 41 \
  --mask-scale-x 0.95 --mask-scale-y 0.90

# =========================
# 5) Start service (FastAPI) – runs in background
# =========================
# ## 1) set envs (edit paths)
export GEN_CKPT=work_dirs/train/ckpts/gen_B_best.pt
export ID_ENC_CKPT=work_dirs/train/ckpts/id_encoder_fixed.pt
export DEVICE=cpu
export MAX_IMAGE_MB=10
export OUTPUT_DIR=work_dirs/services

# ## 2) start server
uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 1

## 3) swap
# curl -v -D - -X POST http://127.0.0.1:8000/swap \
#   -F "src=@$(pwd)/data/assets/src.png" \
#   -F "tgt=@$(pwd)/data/assets/tgt.png" \
#   -F "blend=feather" \
#   -F "feather_ksize=31" \
#   -F "color_match=true" \
#   -o out.png


# =========================
# 6) Video inference
# =========================
python -m scripts.infer_video \
  --src data/assets/src.png \
  --tgt-video data/assets/demo.mp4 \
  --out-video work_dirs/infer/demo_swap.mp4 \
  --ckpt work_dirs/train/ckpts/gen_B_best.pt \
  --id-enc-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt \
  --device cpu \
  --blend feather --feather-ksize 41 --color-match \
  --detect-every 3 --smooth-alpha 0.6 --temporal-alpha 0.75 \
  --mask-scale-x 0.95 --mask-scale-y 0.9

python -m scripts.rt_face_swap \
  --src data/assets/src.png \
  --input 0 \
  --ckpt work_dirs/train/ckpts/gen_B_best.pt \
  --id-enc-ckpt work_dirs/train/ckpts/id_encoder_fixed.pt \
  --device cpu --cpu-threads 4 \
  --detect-every 4 --smooth-alpha 0.65 --temporal-alpha 0.8 \
  --feather-ksize 41 --color-match \
  --queue-size 2 --drop-policy drop_oldest
