from __future__ import annotations
import argparse
from pathlib import Path
import time

from utils.image_io import load_rgb, pil_to_numpy_rgb, save_rgb
from utils.infer_core import InferenceConfig, FaceSwapInference

def parse_args():
    p = argparse.ArgumentParser(description="Face swap inference (image CLI)")
    p.add_argument("--src", type=str, required=True, help="Source image path (identity donor)")
    p.add_argument("--tgt", type=str, required=True, help="Target image path (face receiver)")
    p.add_argument("--ckpt", type=str, default="models/gen_B_best.pt", help="Generator checkpoint")
    p.add_argument("--id-enc-ckpt", type=str, default="", help="Fixed IDEncoder checkpoint saved during training")
    p.add_argument("--out", type=str, default="swap_out.png", help="Output image path")
    p.add_argument("--blend", type=str, default="feather", choices=["feather","poisson"])
    p.add_argument("--feather-ksize", type=int, default=25)
    p.add_argument("--color-match", action="store_true", help="Enable HSV lightness match")
    p.add_argument("--src-index", type=int, default=None, help="Pick Nth detected face in source (default: highest prob)")
    p.add_argument("--tgt-index", type=int, default=None, help="Pick Nth detected face in target (default: highest prob)")
    p.add_argument("--mask-scale-x", type=float, default=1.0, help="Scale ellipse mask horizontally in aligned space")
    p.add_argument("--mask-scale-y", type=float, default=1.0, help="Scale ellipse mask vertically in aligned space")
    return p.parse_args()

def main():
    args = parse_args()

    # Load images
    src = pil_to_numpy_rgb(load_rgb(args.src))
    tgt = pil_to_numpy_rgb(load_rgb(args.tgt))

    cfg = InferenceConfig(
        ckpt_path=args.ckpt,
        id_enc_ckpt_path=args.id_enc_ckpt or None,
        blend_mode=args.blend,
        feather_ksize=args.feather_ksize,
        color_match=bool(args.color_match),
        src_index=args.src_index,
        tgt_index=args.tgt_index,
        mask_scale_x=args.mask_scale_x,
        mask_scale_y=args.mask_scale_y
    )
    engine = FaceSwapInference(cfg)

    t0 = time.time()
    out = engine.swap_images(src, tgt)
    dt = time.time() - t0

    save_rgb(out["composite"], args.out)
    print(f"Saved -> {args.out} | blend={cfg.blend_mode} color_match={cfg.color_match} "
          f"mask_scale=({cfg.mask_scale_x},{cfg.mask_scale_y}) time={dt:.2f}s")
    print(f"Debug: {out['debug']}")

if __name__ == "__main__":
    main()
