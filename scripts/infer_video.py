import argparse
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.infer_core import InferenceConfig, FaceSwapInference
from utils.face_align import detect_face_5pt
from utils.image_io import pil_to_numpy_rgb


def ema_update(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential moving average for 2D landmarks.
    alpha in [0,1]: alpha=0.0 -> follow current strictly; alpha=1.0 -> freeze at previous.
    We interpret 'smooth_alpha' from CLI as EMA weight for previous (inertia).
    """
    if prev is None:
        return cur.copy()
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)

def landmarks_to_ellipse_mask(
    landmarks_5pt: np.ndarray,
    hw: Tuple[int, int],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    feather_ksize: int = 31,
) -> np.ndarray:
    """
    Build a soft elliptical mask around the face using 5-pt landmarks.
    - landmarks_5pt: (5,2) in image coords
    - hw: (H,W)
    - scale_x/y: scale mask size relative to face box
    - feather_ksize: Gaussian blur kernel to soften edges
    """
    H, W = hw
    mask = np.zeros((H, W), dtype=np.uint8)
    # rough face center from eye and nose points
    cx = float(np.mean(landmarks_5pt[:, 0]))
    cy = float(np.mean(landmarks_5pt[:, 1]))
    # rough face width/height from eye distance
    eye_dist = np.linalg.norm(landmarks_5pt[0] - landmarks_5pt[1]) + 1e-6
    rx = int(eye_dist * 1.8 * scale_x)  # horizontal radius
    ry = int(eye_dist * 2.4 * scale_y)  # vertical radius (longer to cover cheeks/chin)
    # draw filled ellipse
    cv2.ellipse(mask, (int(cx), int(cy)), (max(rx,1), max(ry,1)), 0, 0, 360, 255, -1)
    # feather for smooth edges
    k = max(3, feather_ksize | 1)  # odd kernel
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    # normalize to [0,1] float
    return (mask.astype(np.float32) / 255.0)

def temporal_blend(prev_out: np.ndarray, cur_out: np.ndarray, face_mask: np.ndarray, t_alpha: float) -> np.ndarray:
    """
    Cross-frame blending on face region only:
    final = t_alpha * cur_out + (1-t_alpha) * prev_out within mask;
    outside mask -> take cur_out.
    - prev_out, cur_out: uint8 HxWx3
    - face_mask: float HxW in [0,1]
    - t_alpha: per-frame blend factor for CURRENT output inside the mask
    """
    if prev_out is None:
        return cur_out
    fm = face_mask[..., None]  # HxWx1
    cur_f = cur_out.astype(np.float32)
    prev_f = prev_out.astype(np.float32)
    mixed_face = (t_alpha * cur_f + (1.0 - t_alpha) * prev_f)
    # composite: mix inside mask; outside mask use current
    out = (fm * mixed_face + (1.0 - fm) * cur_f).astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser(description="Face-swap video with temporal smoothing.")
    ap.add_argument("--src", required=True, help="Source face image (identity donor)")
    ap.add_argument("--tgt-video", required=True, help="Target video path")
    ap.add_argument("--out-video", required=True, help="Output video path (.mp4 recommended)")
    ap.add_argument("--ckpt", required=True, help="Generator checkpoint")
    ap.add_argument("--id-enc-ckpt", required=True, help="ID encoder checkpoint (from Stage A)")
    ap.add_argument("--device", default="cpu", help="'cuda' or 'cpu'")
    ap.add_argument("--blend", default="feather", choices=["feather", "poisson"])
    ap.add_argument("--feather-ksize", type=int, default=31)
    ap.add_argument("--color-match", action="store_true", help="Enable per-frame color matching")
    ap.add_argument("--mask-scale-x", type=float, default=1.0)
    ap.add_argument("--mask-scale-y", type=float, default=0.95)
    ap.add_argument("--detect-every", type=int, default=3, help="Re-detect landmarks every N frames")
    ap.add_argument("--smooth-alpha", type=float, default=0.6, help="EMA inertia for landmarks [0..1], higher=more stable")
    ap.add_argument("--temporal-alpha", type=float, default=0.75, help="Blend weight for current output [0..1] inside mask")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit number of frames for quick testing (0=all)")
    args = ap.parse_args()

    # Build inference engine (weights load once)
    cfg = InferenceConfig(
        ckpt_path=args.ckpt,
        id_enc_ckpt_path=args.id_enc_ckpt,
        device=args.device,
        blend_mode=args.blend,
        feather_ksize=args.feather_ksize,
        color_match=bool(args.color_match),
        src_index=None,
        tgt_index=None,
        mask_scale_x=args.mask_scale_x,
        mask_scale_y=args.mask_scale_y,
    )
    engine = FaceSwapInference(cfg)

    # Load source image -> numpy RGB
    src_img = Image.open(args.src).convert("RGB")
    src_np = pil_to_numpy_rgb(src_img)

    # Open video
    cap = cv2.VideoCapture(args.tgt_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.tgt_video}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Video writer (mp4v is widely supported)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = Path(args.out_video).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out_video, fourcc, FPS, (W, H))

    # Temporal states
    prev_out = None
    sm_landmarks = None  # smoothed 5-pt landmarks
    frame_idx = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            # BGR -> RGB for inference
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 1) Run face swap for this frame
            result = engine.swap_images(src_np, frame_rgb)
            comp_rgb = result["composite"]  # uint8 HxWx3

            # 2) Build / update landmarks for temporal mask (detect every N frames)
            do_detect = (sm_landmarks is None) or (frame_idx % args.detect_every == 1)
            if do_detect:
                pts = detect_face_5pt(frame_rgb)  # (5,2) or None
                if pts is not None:
                    sm_landmarks = ema_update(sm_landmarks, pts.astype(np.float32), alpha=args.smooth_alpha)
                # if detection failed, keep previous smoothed landmarks (best effort)
            else:
                # Even if we didn't detect, we can slightly decay toward nothing by increasing alpha;
                # but typically keeping last sm_landmarks works fine for short spans.
                pass

            # 3) Build face mask from (smoothed) landmarks
            if sm_landmarks is not None:
                face_mask = landmarks_to_ellipse_mask(
                    sm_landmarks, (H, W),
                    scale_x=args.mask_scale_x, scale_y=args.mask_scale_y,
                    feather_ksize=args.feather_ksize
                )
            else:
                # Fallback: global light feather mask (minimal smoothing)
                face_mask = np.ones((H, W), dtype=np.float32)

            # 4) Temporal blending within face region only
            comp_bgr = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)
            out_frame = temporal_blend(prev_out, comp_bgr, face_mask, t_alpha=args.temporal_alpha)
            prev_out = out_frame

            writer.write(out_frame)

            # (Optional) print progress
            if frame_idx % 50 == 0:
                print(f"[infer_video] processed {frame_idx} frames...")

    finally:
        cap.release()
        writer.release()

    print(f"[infer_video] done, saved -> {args.out_video}")

if __name__ == "__main__":
    main()
