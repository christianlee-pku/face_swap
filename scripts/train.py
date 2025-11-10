import argparse
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from models.id_encoder_mobile import IDEncoderMobile
from models.generator_mobileswaplite import MobileSwapLite
from utils.losses import MobileNetV2Perceptual, cosine_id_loss
from utils.logging_utils import setup_logger, human_bytes, save_plot, CSVLogger
from utils.train_utils import (
    set_seed, count_params, make_loader, filter_pairs_csv,
    current_lr, safe_item
)

def run_one_epoch(
    stage: str,
    epoch: int,
    model_g: nn.Module,
    model_id: nn.Module,
    perceptual: nn.Module,
    dl: DataLoader,
    device: str,
    w_id: float, w_perc: float, w_l1_face: float, w_l1_bg: float,
    optimizer: Optional[optim.Optimizer],
    log_interval: int,
    logger,
    batch_logger: Optional[CSVLogger] = None,
    grad_clip: float = 0.0,
    log_grad_norm: bool = False
):
    is_train = optimizer is not None
    model_g.train() if is_train else model_g.eval()
    torch.set_grad_enabled(is_train)

    n_batches = len(dl)
    meters = {"g":0.0,"id":0.0,"perc":0.0,"l1f":0.0,"l1bg":0.0}
    start_epoch_time = time.time()
    end_iter = time.time()

    for bi, batch in enumerate(dl, start=1):
        data_time = time.time() - end_iter
        tic = time.time()

        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        msk = batch["tgt_mask"].to(device, non_blocking=True)
        same = batch["same"].to(device, non_blocking=True)

        with torch.no_grad():
            emb_src = model_id(src)  # Bx128

        out = model_g(tgt, emb_src)
        emb_out = model_id(out)

        loss_id = cosine_id_loss(emb_out, emb_src)
        loss_perc = perceptual(out, tgt)
        l1map = torch.abs(out - tgt)
        loss_l1_face = (l1map * msk).mean()
        loss_l1_bg   = (l1map * (1.0 - msk)).mean()

        if stage.upper() == "A":
            coef = (same.float().mean() + 1e-6).item()
            loss_l1_face = loss_l1_face * (1.0 + coef)
            loss_l1_bg   = loss_l1_bg   * (1.0 + coef)

        loss_total = (w_id*loss_id + w_perc*loss_perc +
                      w_l1_face*loss_l1_face + w_l1_bg*loss_l1_bg)

        grad_norm_val = 0.0
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            if log_grad_norm or grad_clip > 0:
                total_norm = torch.norm(torch.stack([
                    p.grad.detach().data.norm(2)
                    for p in model_g.parameters() if p.grad is not None
                ]), 2.0)
                grad_norm_val = safe_item(total_norm)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model_g.parameters(), grad_clip)
            optimizer.step()

        iter_time = time.time() - tic

        meters["g"]   += safe_item(loss_total)
        meters["id"]  += safe_item(loss_id)
        meters["perc"]+= safe_item(loss_perc)
        meters["l1f"] += safe_item(loss_l1_face)
        meters["l1bg"]+= safe_item(loss_l1_bg)

        mem_alloc = mem_resv = 0
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated()
            mem_resv  = torch.cuda.memory_reserved()

        if (bi % log_interval == 0) or (bi == n_batches):
            lr = current_lr(optimizer) if is_train else 0.0
            logger.info(
                f"[{stage}] ep {epoch:03d} | batch {bi:04d}/{n_batches:04d} | "
                f"data {data_time*1000:.1f} ms | iter {iter_time*1000:.1f} ms | "
                f"lr {lr:.6f} | "
                f"loss: total={safe_item(loss_total):.4f} id={safe_item(loss_id):.4f} "
                f"perc={safe_item(loss_perc):.4f} l1f={safe_item(loss_l1_face):.4f} "
                f"l1bg={safe_item(loss_l1_bg):.4f} | "
                f"grad {grad_norm_val:.2f} | "
                f"mem {human_bytes(mem_alloc)}/{human_bytes(mem_resv)}"
            )

        if batch_logger is not None:
            batch_logger.write({
                "stage":stage, "epoch":epoch, "batch":bi, "num_batches":n_batches,
                "lr": current_lr(optimizer) if is_train else 0.0,
                "data_time": data_time, "iter_time": iter_time,
                "loss_total": safe_item(loss_total), "loss_id": safe_item(loss_id),
                "loss_perc": safe_item(loss_perc), "loss_l1_face": safe_item(loss_l1_face),
                "loss_l1_bg": safe_item(loss_l1_bg),
                "grad_norm": grad_norm_val,
                "cuda_mem_alloc": mem_alloc, "cuda_mem_resv": mem_resv
            })

        end_iter = time.time()

    n = max(1, len(dl))
    for k in meters: meters[k] /= n
    epoch_time = time.time() - start_epoch_time
    return meters, epoch_time

def train_stage(
    stage: str,
    train_csv: str, val_csv: Optional[str],
    epochs: int,
    batch_size: int,
    lr: float,
    w_id: float, w_perc: float, w_l1_face: float, w_l1_bg: float,
    same_only: bool,
    out_dir: Path,
    num_workers: int,
    base: int,
    id_dim: int,
    width_id: float,
    grad_clip: float,
    log_interval: int,
    log_grad_norm: bool,
    logger,
    init_gen: str = "",            
    id_enc_ckpt: str = "",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    train_csv_eff = filter_pairs_csv(train_csv, same_only=same_only)
    ds_tr, dl_tr = make_loader(train_csv_eff, batch_size, aug=True,  workers=num_workers)
    ds_va, dl_va = (None, None)
    if val_csv:
        ds_va, dl_va = make_loader(val_csv, batch_size, aug=False, workers=num_workers)

    logger.info(f"[{stage}] Data prepared: train={len(ds_tr)} samples, "
                f"val={len(ds_va) if ds_va else 0} samples, batch_size={batch_size}")

    # models: ID encoder
    id_enc = IDEncoderMobile(width=width_id, emb_dim=id_dim).to(device).eval()
    if id_enc_ckpt:
        sd = torch.load(id_enc_ckpt, map_location=device)
        sd = sd.get("state_dict", sd)
        id_enc.load_state_dict(sd, strict=True)
        logger.info(f"[{stage}] Loaded IDEncoder from {id_enc_ckpt}")
    else:
        if stage.upper() == "A":
            save_fp = out_dir / "id_encoder_fixed.pt"
            if not save_fp.exists():
                torch.save({"state_dict": id_enc.state_dict(), "width": width_id, "emb_dim": id_dim}, save_fp)
                logger.info(f"[{stage}] Saved fixed IDEncoder -> {save_fp.as_posix()}")
    for p in id_enc.parameters(): 
        p.requires_grad = False

    # models: Generator
    gen = MobileSwapLite(base=base, id_dim=id_dim).to(device)
    if init_gen and stage.upper() == "B":
        gsd = torch.load(init_gen, map_location=device)
        gsd = gsd.get("gen", gsd)
        gen.load_state_dict(gsd, strict=True)
        logger.info(f"[{stage}] Initialized generator from {init_gen}")

    logger.info(f"[{stage}] IDEncoderMobile params: {count_params(id_enc):,}")
    logger.info(f"[{stage}] MobileSwapLite params: {count_params(gen):,}")

    # optim & losses
    opt_g = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    percept = MobileNetV2Perceptual(layers=(2,4,7,14), reduction="l1", normalize=False).to(device)

    # CSV loggers
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    batch_logger = CSVLogger(
        logs_dir / f"metrics_batch_{stage}.csv",
        fieldnames=["stage","epoch","batch","num_batches","lr","data_time","iter_time",
                    "loss_total","loss_id","loss_perc","loss_l1_face","loss_l1_bg",
                    "grad_norm","cuda_mem_alloc","cuda_mem_resv"]
    )
    epoch_logger = CSVLogger(
        logs_dir / f"metrics_epoch_{stage}.csv",
        fieldnames=["stage","epoch","time_epoch",
                    "train_g","train_id","train_perc","train_l1f","train_l1bg",
                    "val_g","val_id","val_perc","val_l1f","val_l1bg"]
    )

    history = {"g":[], "id":[], "perc":[], "l1f":[], "l1bg":[]}
    best_val_id = None

    for ep in range(1, epochs+1):
        logger.info(f"[{stage}] ===== Epoch {ep}/{epochs} START =====")
        logger.info(f"[{stage}] LR = {current_lr(opt_g):.6f} | grad_clip = {grad_clip} | device = {device}")

        train_metrics, t_tr = run_one_epoch(
            stage=stage, epoch=ep, model_g=gen, model_id=id_enc, perceptual=percept,
            dl=dl_tr, device=device,
            w_id=w_id, w_perc=w_perc, w_l1_face=w_l1_face, w_l1_bg=w_l1_bg,
            optimizer=opt_g, log_interval=log_interval, logger=logger,
            batch_logger=batch_logger, grad_clip=grad_clip, log_grad_norm=log_grad_norm
        )
        logger.info(f"[{stage}] Train@Epoch{ep}: "
                    f"g={train_metrics['g']:.4f} id={train_metrics['id']:.4f} "
                    f"perc={train_metrics['perc']:.4f} l1f={train_metrics['l1f']:.4f} "
                    f"l1bg={train_metrics['l1bg']:.4f} | time={t_tr:.1f}s")

        val_metrics = {"g": float("nan"), "id": float("nan"), "perc": float("nan"),
                       "l1f": float("nan"), "l1bg": float("nan")}
        t_va = 0.0
        if dl_va is not None and len(dl_va) > 0:
            val_metrics, t_va = run_one_epoch(
                stage=stage, epoch=ep, model_g=gen, model_id=id_enc, perceptual=percept,
                dl=dl_va, device=device,
                w_id=w_id, w_perc=w_perc, w_l1_face=w_l1_face, w_l1_bg=w_l1_bg,
                optimizer=None, log_interval=max(1, log_interval//2), logger=logger,
                batch_logger=None, grad_clip=0.0, log_grad_norm=False
            )
            logger.info(f"[{stage}] Val@Epoch{ep}:   "
                        f"g={val_metrics['g']:.4f} id={val_metrics['id']:.4f} "
                        f"perc={val_metrics['perc']:.4f} l1f={val_metrics['l1f']:.4f} "
                        f"l1bg={val_metrics['l1bg']:.4f} | time={t_va:.1f}s")

        for k in history: history[k].append(train_metrics[k])

        epoch_logger.write({
            "stage":stage, "epoch":ep, "time_epoch":round(t_tr + t_va, 2),
            "train_g":train_metrics["g"], "train_id":train_metrics["id"],
            "train_perc":train_metrics["perc"], "train_l1f":train_metrics["l1f"], "train_l1bg":train_metrics["l1bg"],
            "val_g":val_metrics["g"], "val_id":val_metrics["id"],
            "val_perc":val_metrics["perc"], "val_l1f":val_metrics["l1f"], "val_l1bg":val_metrics["l1bg"]
        })

        ckpt_path = out_dir / f"gen_{stage}_ep{ep}.pt"
        torch.save({"gen": gen.state_dict()}, ckpt_path)
        logger.info(f"[{stage}] Checkpoint saved -> {ckpt_path.as_posix()}")

        if dl_va is not None and not math.isnan(val_metrics["id"]):
            score = val_metrics["id"]
            if (best_val_id is None) or (score < best_val_id):
                best_val_id = score
                best_path = out_dir / f"gen_{stage}_best.pt"
                torch.save({"gen": gen.state_dict()}, best_path)
                logger.info(f"[{stage}] New BEST (val id={best_val_id:.4f}) -> {best_path.as_posix()}")

        save_plot(history, out_dir / f"loss_{stage}.png", f"Losses Stage {stage}", logger)
        logger.info(f"[{stage}] ===== Epoch {ep}/{epochs} END =====")

    batch_logger.close(); epoch_logger.close()
    last_path = out_dir / f"gen_{stage}_last.pt"
    torch.save({"gen": gen.state_dict()}, last_path)
    logger.info(f"[{stage}] DONE. Last checkpoint -> {last_path.as_posix()}")

def parse_args():
    p = argparse.ArgumentParser(description="MobileSwapLite training (training core kept in train.py)")
    # Data
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--val-csv",   type=str, default="")
    p.add_argument("--num-workers", type=int, default=2)
    # Stages
    p.add_argument("--stageA-epochs", type=int, default=6)
    p.add_argument("--stageA-same-only", action="store_true")
    p.add_argument("--stageB-epochs", type=int, default=18)
    # Hyperparams
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    # Loss weights
    p.add_argument("--w-id", type=float, default=1.0)
    p.add_argument("--w-perc", type=float, default=0.2)
    p.add_argument("--w-l1-face", type=float, default=0.2)
    p.add_argument("--w-l1-bg", type=float, default=1.0)
    # Model sizes
    p.add_argument("--base", type=int, default=24)
    p.add_argument("--id-dim", type=int, default=128)
    p.add_argument("--width-id", type=float, default=1.0)
    # Init / checkpoints (NEW)
    p.add_argument("--init-gen", type=str, default="", help="Init generator from given checkpoint (applied to Stage B)")
    p.add_argument("--id-enc-ckpt", type=str, default="", help="Load fixed IDEncoder checkpoint saved in Stage A")
    # Logs & output
    p.add_argument("--out-dir", type=str, default="ckpts")
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--log-grad-norm", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else (out_dir / "train.log")
    logger = setup_logger(log_file, level=args.log_level)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("==== Environment ====")
    logger.info(f"PyTorch {torch.__version__} | TorchVision {torchvision.__version__} | CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Seed={args.seed} | Device={device}")
    logger.info("==== HParams ====")
    logger.info(
        f"train_csv={args.train_csv} | val_csv={args.val_csv or '(none)'} | "
        f"batch_size={args.batch_size} | lr={args.lr} | workers={args.num_workers} | "
        f"w_id={args.w_id} w_perc={args.w_perc} w_l1_face={args.w_l1_face} w_l1_bg={args.w_l1_bg} | "
        f"base={args.base} id_dim={args.id_dim} width_id={args.width_id}"
    )
    logger.info(
        f"stageA_epochs={args.stageA_epochs} same_only={args.stageA_same_only} | "
        f"stageB_epochs={args.stageB_epochs} | grad_clip={args.grad_clip} | "
        f"log_interval={args.log_interval} | log_grad_norm={args.log_grad_norm}"
    )
    logger.info(f"init_gen={args.init_gen or '(none)'} | id_enc_ckpt={args.id_enc_ckpt or '(none)'}")
    logger.info(f"Logs -> {log_file.as_posix()} | Outputs -> {out_dir.as_posix()}")

    # Stage A
    if args.stageA_epochs > 0:
        train_stage(
            stage="A",
            train_csv=args.train_csv,
            val_csv=args.val_csv if args.val_csv else None,
            epochs=args.stageA_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            w_id=args.w_id, w_perc=args.w_perc, w_l1_face=args.w_l1_face, w_l1_bg=args.w_l1_bg,
            same_only=args.stageA_same_only,
            out_dir=out_dir, num_workers=args.num_workers,
            base=args.base, id_dim=args.id_dim, width_id=args.width_id,
            grad_clip=args.grad_clip, log_interval=args.log_interval,
            log_grad_norm=args.log_grad_norm, logger=logger,
            init_gen="", id_enc_ckpt=args.id_enc_ckpt
        )

    # Stage B
    if args.stageB_epochs > 0:
        train_stage(
            stage="B",
            train_csv=args.train_csv,
            val_csv=args.val_csv if args.val_csv else None,
            epochs=args.stageB_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            w_id=args.w_id, w_perc=args.w_perc, w_l1_face=args.w_l1_face, w_l1_bg=args.w_l1_bg,
            same_only=False,
            out_dir=out_dir, num_workers=args.num_workers,
            base=args.base, id_dim=args.id_dim, width_id=args.width_id,
            grad_clip=args.grad_clip, log_interval=args.log_interval,
            log_grad_norm=args.log_grad_norm, logger=logger,
            init_gen=args.init_gen, id_enc_ckpt=args.id_enc_ckpt
        )

    logger.info("All done.")

if __name__ == "__main__":
    main()
