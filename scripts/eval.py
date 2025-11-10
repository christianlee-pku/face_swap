# Evaluate one or many checkpoints on the validation CSV and plot accuracy curves.
#
# Usage example:
#   python -m scripts.eval \
#     --val-csv data/metadata/pairs_val.csv \
#     --ckpt-glob "models/gen_B_ep*.pt" \
#     --batch-size 8 --out-dir reports
#
from __future__ import annotations
import argparse
import glob
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.id_encoder_mobile import IDEncoderMobile
from models.generator_mobileswaplite import MobileSwapLite
from utils.train_utils import make_loader, set_seed
from utils.logging_utils import setup_logger
from eval.metrics import to01, cos_sim, ssim, ms_ssim, psnr, psnr_masked

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 Evaluation & Accuracy Graphs")
    p.add_argument("--val-csv", type=str, required=True, help="Validation pairs CSV")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=str, help="Single checkpoint path")
    g.add_argument("--ckpt-glob", type=str, help='Glob pattern, e.g. "ckpts/gen_B_ep*.pt"')
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="Optional limit of samples for quick run")
    p.add_argument("--out-dir", type=str, default="reports")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()

def _collect_ckpts(args) -> List[Path]:
    if args.ckpt:
        return [Path(args.ckpt)]
    paths = [Path(p) for p in glob.glob(args.ckpt_glob)]
    paths.sort(key=lambda p: _epoch_from_name(p.name))
    return paths

_epoch_re = re.compile(r"_ep(\d+)\.pt$")
def _epoch_from_name(name: str) -> int:
    m = _epoch_re.search(name)
    return int(m.group(1)) if m else -1

@torch.no_grad()
def _embed(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return encoder(x)

def evaluate_checkpoint(ckpt_path: Path,
                        val_csv: str,
                        batch_size: int,
                        num_workers: int,
                        limit: int,
                        device: str,
                        out_dir: Path,
                        log_level: str = "INFO") -> Dict[str, float]:
    # per-ckpt logger
    log_file = out_dir / f"eval_{ckpt_path.stem}.log"
    logger = setup_logger(log_file, level=log_level)
    logger.info(f"Evaluating checkpoint: {ckpt_path.as_posix()}")

    # data
    ds, dl = make_loader(val_csv, batch_size=batch_size, aug=False, workers=num_workers)
    if limit > 0:
        # create a limited sampler by slicing the dataset indices
        from torch.utils.data import Subset
        N = min(limit, len(ds))
        ds = Subset(ds, list(range(N)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    logger.info(f"Val samples: {len(ds)} | batch_size={batch_size} | device={device}")

    # models
    id_enc = IDEncoderMobile().to(device).eval()
    gen = MobileSwapLite().to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    gen.load_state_dict(state["gen"], strict=True)
    logger.info("Models loaded.")

    # accumulators
    id_cos_all, ssim_all, msssim_all, psnr_all = [], [], [], []
    psnr_face_all, psnr_bg_all = [], []

    t0 = time.time()
    for bi, batch in enumerate(dl, start=1):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        msk = batch["tgt_mask"].to(device)

        emb_src = _embed(id_enc, src)
        out = gen(tgt, emb_src)
        emb_out = _embed(id_enc, out)

        # metrics
        # Identity preservation
        id_cos_batch = cos_sim(emb_out, emb_src).detach().cpu().numpy()  # per-sample
        id_cos_all.extend(id_cos_batch.tolist())

        # structural fidelity
        out01, tgt01 = to01(out), to01(tgt)
        ssim_val = float(ssim(out01, tgt01).detach().cpu())
        ssim_all.append(ssim_val)

        ms_val = ms_ssim(out01, tgt01)
        if ms_val is not None:
            msssim_all.append(float(ms_val.detach().cpu()))

        psnr_all.append(float(psnr(out01, tgt01).detach().cpu()))
        psnr_face_all.append(float(psnr_masked(out01, tgt01, msk).detach().cpu()))
        psnr_bg_all.append(float(psnr_masked(out01, tgt01, 1.0 - msk).detach().cpu()))

        if bi % 20 == 0:
            logger.info(f"batch {bi}/{len(dl)}: "
                        f"IDcos={np.mean(id_cos_batch):.4f} | "
                        f"SSIM={ssim_val:.4f} | PSNR={psnr_all[-1]:.2f} dB")

    t1 = time.time()
    logger.info(f"Done. Elapsed {t1 - t0:.1f}s")

    # aggregate
    res = {
        "ckpt": ckpt_path.name,
        "epoch": _epoch_from_name(ckpt_path.name),
        "IDcos_mean": float(np.mean(id_cos_all)) if id_cos_all else float("nan"),
        "IDcos_std":  float(np.std(id_cos_all)) if id_cos_all else float("nan"),
        "SSIM_mean":  float(np.mean(ssim_all))  if ssim_all else float("nan"),
        "PSNR_mean":  float(np.mean(psnr_all))  if psnr_all else float("nan"),
        "PSNR_face_mean": float(np.nanmean(psnr_face_all)) if psnr_face_all else float("nan"),
        "PSNR_bg_mean":   float(np.nanmean(psnr_bg_all))   if psnr_bg_all else float("nan"),
        "MSSSIM_mean": float(np.mean(msssim_all)) if msssim_all else float("nan"),
        "num_samples": int(len(ds)),
        "time_sec": float(t1 - t0),
    }
    logger.info("Summary: " + " | ".join([f"{k}={v}" for k,v in res.items() if k not in ("ckpt",)]))
    return res

def plot_curves(csv_path: Path, save_path: Path):
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    if "epoch" in df.columns:
        df = df.sort_values("epoch")
    x = df["epoch"] if "epoch" in df.columns else range(len(df))
    plt.figure(figsize=(9,5))
    keys = [k for k in ["IDcos_mean","SSIM_mean","MSSSIM_mean","PSNR_mean","PSNR_face_mean","PSNR_bg_mean"] if k in df.columns]
    for k in keys:
        plt.plot(x, df[k], marker="o", label=k)
    plt.xlabel("epoch"); plt.title("Phase 3 Accuracy Curves"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "phase3_eval.csv"

    ckpts = _collect_ckpts(args)
    if not ckpts:
        raise SystemExit("No checkpoints found. Check --ckpt or --ckpt-glob.")

    all_rows: List[Dict] = []
    for ck in ckpts:
        row = evaluate_checkpoint(
            ckpt_path=ck,
            val_csv=args.val_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.limit,
            device=device,
            out_dir=out_dir,
            log_level=args.log_level
        )
        all_rows.append(row)
        # append to CSV incrementally
        df = pd.DataFrame(all_rows)
        df.to_csv(master_csv, index=False)

    # curves
    try:
        plot_curves(master_csv, out_dir / "phase3_curves.png")
    except Exception as e:
        print(f"Could not plot curves: {e}")

if __name__ == "__main__":
    main()
