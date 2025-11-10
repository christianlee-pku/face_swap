# Evaluate face-swap quality and create accuracy graphs.
# - Computes identity cosine metrics and PSNR on (out, tgt)
# - Saves per-sample CSV and multiple diagnostic plots
# - No external deps beyond numpy/matplotlib/sklearn

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc

import torch
from utils.infer_core import InferenceConfig, FaceSwapInference


def read_pairs_csv(csv_path):
    """
    Read a pairs CSV. Tries common column names:
      src | src_path, tgt | tgt_path, same | same_identity
    Returns a list of dicts: {src, tgt, same}
    """
    import csv
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rr = {k.lower(): v for k, v in r.items()}
            src = rr.get("src") or rr.get("src_path")
            tgt = rr.get("tgt") or rr.get("tgt_path")
            same = rr.get("same") or rr.get("same_identity") or "0"
            same = int(str(same).strip()) if str(same).strip() != "" else 0
            rows.append({"src": src, "tgt": tgt, "same": same})
    return rows

def pil_to_np_rgb(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"))

@torch.no_grad()
def embed_id(encoder: torch.nn.Module, img_np: np.ndarray) -> torch.Tensor:
    """
    Encode an RGB HxWx3 uint8 array to an L2-normalized embedding (1 x D).
    Assume the ID encoder in engine expects [-1,1] normalized tensors.
    """
    x = torch.tensor(img_np).float() / 255.0
    x = (x - 0.5) / 0.5
    x = x.permute(2, 0, 1).unsqueeze(0)
    e = encoder(x)
    e = e / (e.norm(dim=1, keepdim=True) + 1e-8)
    return e

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio for uint8 RGB images.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))

def save_hist(values, out_path: Path, title: str, xlabel: str):
    plt.figure()
    plt.hist(values, bins=50)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path.as_posix())
    plt.close()

def save_scatter(xv, yv, out_path: Path, title: str, xlabel: str, ylabel: str):
    plt.figure()
    plt.scatter(xv, yv, s=6, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path.as_posix())
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Evaluate face swap accuracy and create graphs.")
    ap.add_argument("--val-csv", required=True, help="Validation pairs CSV")
    ap.add_argument("--ckpt", required=True, help="Generator checkpoint")
    ap.add_argument("--id-enc-ckpt", required=True, help="ID encoder checkpoint (fixed from Stage A)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="work_dirs/eval")
    ap.add_argument("--max-samples", type=int, default=1000, help="Cap the number of pairs to evaluate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build inference engine
    cfg = InferenceConfig(
        ckpt_path=args.ckpt,
        id_enc_ckpt_path=args.id_enc_ckpt,
        device=args.device,
        blend_mode="feather",
        feather_ksize=31,
        color_match=False,
        src_index=None,
        tgt_index=None,
        mask_scale_x=1.0,
        mask_scale_y=1.0,
    )
    engine = FaceSwapInference(cfg)
    E = engine.id_enc  # frozen ID encoder

    # Load validation pairs
    pairs = read_pairs_csv(args.val_csv)
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    # Iterate and collect metrics
    rows = []
    cos_src_list, cos_tgt_list, psnr_list, label_same = [], [], [], []
    for i, it in enumerate(pairs, 1):
        p_src = Path(it["src"])
        p_tgt = Path(it["tgt"])
        if not p_src.exists() or not p_tgt.exists():
            # skip silently if paths are invalid
            continue
        src_np = pil_to_np_rgb(Image.open(p_src))
        tgt_np = pil_to_np_rgb(Image.open(p_tgt))

        # Run swap
        out = engine.swap_images(src_np, tgt_np)
        comp_np = out["composite"]

        # Embeddings
        e_src = embed_id(E, src_np)
        e_tgt = embed_id(E, tgt_np)
        e_out = embed_id(E, comp_np)

        # Cosines
        cos_src = float((e_out * e_src).sum())
        cos_tgt = float((e_out * e_tgt).sum())
        p_val = psnr(comp_np, tgt_np)

        cos_src_list.append(cos_src)
        cos_tgt_list.append(cos_tgt)
        psnr_list.append(p_val)
        label_same.append(int(it["same"]))

        rows.append({
            "idx": i,
            "src": p_src.as_posix(),
            "tgt": p_tgt.as_posix(),
            "same": int(it["same"]),
            "cos_out_src": cos_src,
            "cos_out_tgt": cos_tgt,
            "id_margin": cos_src - cos_tgt,
            "psnr_out_tgt": p_val
        })

    # Save per-sample CSV
    import csv
    csv_path = out_dir / "eval_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # Plots
    save_hist(cos_src_list, out_dir / "idcos_hist.png", "cos(out, src) distribution", "cos")
    margin = (np.array(cos_src_list) - np.array(cos_tgt_list)).tolist()
    save_hist(margin, out_dir / "id_margin_hist.png", "Identity margin: cos(out,src)-cos(out,tgt)", "margin")
    save_scatter(cos_src_list, cos_tgt_list, out_dir / "id_scatter.png",
                 "Identity scatter", "cos(out,src)", "cos(out,tgt)")
    save_hist(psnr_list, out_dir / "psnr_hist.png", "PSNR(out, tgt) distribution", "PSNR (dB)")

    # ROC-style diagnostic for margin separating src vs tgt identity
    # build a pseudo task: higher (cos_out_src - cos_out_tgt) => better "belongs to src"
    y_score = np.array(margin)
    # For labels, use (1 - same) to enforce cross-identity should have large positive margin.
    # If val set has both same/diff, you will get a meaningful AUC.
    y_true = 1 - np.array(label_same, dtype=int)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = float(auc(fpr, tpr))
    except Exception:
        fpr, tpr, auc_val = [], [], float("nan")

    # Save summary JSON
    summary = {
        "num_samples": len(rows),
        "cos_out_src_mean": float(np.mean(cos_src_list)) if cos_src_list else float("nan"),
        "cos_out_tgt_mean": float(np.mean(cos_tgt_list)) if cos_tgt_list else float("nan"),
        "id_margin_mean": float(np.mean(margin)) if margin else float("nan"),
        "psnr_mean": float(np.mean(psnr_list)) if psnr_list else float("nan"),
        "roc_auc_margin": auc_val
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", out_dir.as_posix(), "\nSummary:", summary)

if __name__ == "__main__":
    main()
