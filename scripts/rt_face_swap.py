import argparse
import os
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from utils.infer_core import InferenceConfig, FaceSwapInference
from utils.face_align import detect_face_5pt
from utils.image_io import pil_to_numpy_rgb


def configure_perf(device: str, cpu_threads: int = 4, try_jit: bool = False, engine: Optional[FaceSwapInference] = None):
    """
    Configure runtime performance knobs for CPU/CUDA.
    This version intentionally contains no try/except. If a call fails, it will raise.
    """
    cv2.setNumThreads(1)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # Enable FP16 on generator if available
        if hasattr(engine, "enable_fp16"):
            engine.enable_fp16(True)
        elif hasattr(engine, "gen"):
            engine.gen.half()
        # Optional light JIT optimization
        if try_jit and hasattr(engine, "gen"):
            torch.jit.optimize_for_inference(engine.gen)
    else:
        torch.set_num_threads(max(1, cpu_threads))

def ema_update(prev: Optional[np.ndarray], cur: np.ndarray, alpha_prev: float) -> np.ndarray:
    """
    Exponential moving average for 5-pt landmarks.
    alpha_prev in [0,1] is the weight of PREVIOUS landmarks (higher = more inertia).
    """
    if prev is None:
        return cur.astype(np.float32)
    return (alpha_prev * prev + (1.0 - alpha_prev) * cur).astype(np.float32)


def landmarks_to_ellipse_mask(
    landmarks_5pt: np.ndarray,
    hw: Tuple[int, int],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    feather_ksize: int = 31,
) -> np.ndarray:
    """
    Build a soft elliptical mask around the face.
    - landmarks_5pt: (5,2) in image coords
    - hw: (H,W) of the frame
    """
    H, W = hw
    mask = np.zeros((H, W), dtype=np.uint8)
    cx = float(np.mean(landmarks_5pt[:, 0]))
    cy = float(np.mean(landmarks_5pt[:, 1]))
    eye_dist = np.linalg.norm(landmarks_5pt[0] - landmarks_5pt[1]) + 1e-6
    rx = int(eye_dist * 1.8 * scale_x)
    ry = int(eye_dist * 2.4 * scale_y)
    cv2.ellipse(mask, (int(cx), int(cy)), (max(1, rx), max(1, ry)), 0, 0, 360, 255, -1)
    k = max(3, feather_ksize | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (mask.astype(np.float32) / 255.0)


def temporal_blend(prev_bgr: Optional[np.ndarray], cur_bgr: np.ndarray, face_mask: np.ndarray, alpha_cur: float = 0.8) -> np.ndarray:
    """
    Blend current frame with previous frame only inside the face mask:
    out = mask * (alpha*cur + (1-alpha)*prev) + (1-mask)*cur
    """
    if prev_bgr is None:
        return cur_bgr
    fm = face_mask[..., None]
    cur_f = cur_bgr.astype(np.float32)
    prev_f = prev_bgr.astype(np.float32)
    mixed = alpha_cur * cur_f + (1.0 - alpha_cur) * prev_f
    out = (fm * mixed + (1.0 - fm) * cur_f).astype(np.uint8)
    return out

@dataclass
class RTConfig:
    src_img_path: str
    input_uri: str              # webcam index ("0") or video path
    out_path: Optional[str]     # optional: write result video
    device: str = "cpu"
    ckpt: str = "work_dirs/train/ckpts/gen_B_best.pt"
    id_enc_ckpt: Optional[str] = "work_dirs/train/ckpts/id_encoder_fixed.pt"
    detect_every: int = 3
    smooth_alpha_prev: float = 0.6
    temporal_alpha_cur: float = 0.8
    blend: str = "feather"
    feather_ksize: int = 41
    color_match: bool = True
    mask_scale_x: float = 0.95
    mask_scale_y: float = 0.90
    cpu_threads: int = 4
    try_jit: bool = False
    queue_size: int = 2         # small queue keeps latency low
    drop_policy: str = "drop_oldest"  # or "drop_new"


class RealTimeFaceSwapper:
    def __init__(self, cfg: RTConfig):
        self.cfg = cfg

        # Build engine
        self.engine = FaceSwapInference(
            InferenceConfig(
                ckpt_path=cfg.ckpt,
                id_enc_ckpt_path=cfg.id_enc_ckpt,
                device=cfg.device,
                blend_mode=cfg.blend,
                feather_ksize=cfg.feather_ksize,
                color_match=cfg.color_match,
                src_index=None,
                tgt_index=None,
                mask_scale_x=cfg.mask_scale_x,
                mask_scale_y=cfg.mask_scale_y,
            )
        )
        configure_perf(cfg.device, cfg.cpu_threads, cfg.try_jit, self.engine)

        # Load src image once
        self.src_rgb = pil_to_numpy_rgb(Image.open(cfg.src_img_path).convert("RGB"))

        # Runtime state
        self.cap = None
        self.writer = None
        self.sm_landmarks = None
        self.prev_out = None
        self.running = False

        # Queues
        self.q_in = queue.Queue(maxsize=cfg.queue_size)
        self.q_out = queue.Queue(maxsize=cfg.queue_size)

    def _t_capture(self):
        """
        Capture frames and put into input queue with explicit drop policy if full.
        No try/except: checks queue state before operations.
        """
        # Open capture from index or file path
        if self.cfg.input_uri.isdigit():
            cam_idx = int(self.cfg.input_uri)
            self.cap = cv2.VideoCapture(cam_idx)
        else:
            self.cap = cv2.VideoCapture(self.cfg.input_uri)

        # Ensure opened
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open input: {self.cfg.input_uri}")

        while self.running:
            ok, frame_bgr = self.cap.read()
            if not ok:
                break
            ts = time.time()
            if not self.q_in.full():
                self.q_in.put_nowait((ts, frame_bgr))
            else:
                if self.cfg.drop_policy == "drop_oldest":
                    if not self.q_in.empty():
                        _ = self.q_in.get_nowait()
                    if not self.q_in.full():
                        self.q_in.put_nowait((ts, frame_bgr))
                # else: drop_new -> do nothing (skip this frame)

        self.running = False

    def _t_infer(self):
        """
        Consume frames from input queue, perform swap, build temporal mask with detect-every-N and EMA,
        then push to output queue (drop if full based on policy).
        """
        frame_idx = 0
        H = W = None
        fps_win = []
        t_last = time.time()

        while self.running:
            ts, frame_bgr = self.q_in.get()  # blocking wait
            frame_idx += 1

            if H is None:
                H, W = frame_bgr.shape[:2]

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 1) swap
            res = self.engine.swap_images(self.src_rgb, frame_rgb)
            cur_bgr = cv2.cvtColor(res["composite"], cv2.COLOR_RGB2BGR)

            # 2) detect-every-N and EMA
            if (self.sm_landmarks is None) or (frame_idx % self.cfg.detect_every == 1):
                pts = detect_face_5pt(frame_rgb)  # None or (5,2)
                if pts is not None:
                    self.sm_landmarks = ema_update(self.sm_landmarks, pts.astype(np.float32), self.cfg.smooth_alpha_prev)

            if self.sm_landmarks is not None:
                face_mask = landmarks_to_ellipse_mask(
                    self.sm_landmarks, (H, W),
                    scale_x=self.cfg.mask_scale_x, scale_y=self.cfg.mask_scale_y,
                    feather_ksize=self.cfg.feather_ksize
                )
            else:
                face_mask = np.ones((H, W), dtype=np.float32)

            # 3) temporal blend
            out_bgr = temporal_blend(self.prev_out, cur_bgr, face_mask, alpha_cur=self.cfg.temporal_alpha_cur)
            self.prev_out = out_bgr

            # 4) FPS HUD
            now = time.time()
            dt = max(1e-6, (now - t_last))
            fps = 1.0 / dt
            fps_win.append(fps)
            if len(fps_win) > 30:
                fps_win = fps_win[-30:]
            t_last = now
            cv2.putText(out_bgr, f"FPS:{float(np.mean(fps_win)):.1f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 230, 20), 2)

            # 5) push to output queue
            if not self.q_out.full():
                self.q_out.put_nowait((ts, out_bgr))
            else:
                if self.cfg.drop_policy == "drop_oldest":
                    if not self.q_out.empty():
                        _ = self.q_out.get_nowait()
                    if not self.q_out.full():
                        self.q_out.put_nowait((ts, out_bgr))
                # else: drop_new -> skip

        self.running = False

    def _t_sink(self):
        """
        Display frames and optionally write to a file. No try/except.
        """
        win = "FaceSwap RT"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        out_init = False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while self.running or (not self.q_out.empty()):
            ts, frame_bgr = self.q_out.get()  # blocking wait

            if self.cfg.out_path and not out_init:
                H, W = frame_bgr.shape[:2]
                if len(self.cfg.out_path) > 0:
                    out_dir = os.path.dirname(self.cfg.out_path)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                self.writer = cv2.VideoWriter(self.cfg.out_path, fourcc, 25.0, (W, H))
                out_init = True

            cv2.imshow(win, frame_bgr)
            if self.writer:
                self.writer.write(frame_bgr)

            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC to quit
                self.running = False
                break

        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

    def run(self):
        self.running = True
        th_cap = threading.Thread(target=self._t_capture, daemon=True)
        th_inf = threading.Thread(target=self._t_infer, daemon=True)
        th_out = threading.Thread(target=self._t_sink, daemon=True)
        th_cap.start(); th_inf.start(); th_out.start()

        # Main loop waits until running flag flips (ESC) or capture ends
        while self.running:
            time.sleep(0.1)

        th_cap.join(timeout=1.0)
        th_inf.join(timeout=1.0)
        th_out.join(timeout=1.0)
        if self.cap:
            self.cap.release()


def main():
    ap = argparse.ArgumentParser(description="Real-time face swap (no try/except).")
    ap.add_argument("--src", required=True, help="Source identity image")
    ap.add_argument("--input", required=True, help="Webcam index (e.g., '0') or path to a video file")
    ap.add_argument("--out", default=None, help="Optional output video path (mp4)")
    ap.add_argument("--ckpt", required=True, help="Generator checkpoint")
    ap.add_argument("--id-enc-ckpt", required=True, help="ID encoder checkpoint")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compute device")
    ap.add_argument("--detect-every", type=int, default=3)
    ap.add_argument("--smooth-alpha", type=float, default=0.6, help="EMA weight of previous landmarks [0..1]")
    ap.add_argument("--temporal-alpha", type=float, default=0.8, help="Blend weight for current inside mask [0..1]")
    ap.add_argument("--feather-ksize", type=int, default=41)
    ap.add_argument("--color-match", action="store_true")
    ap.add_argument("--mask-scale-x", type=float, default=0.95)
    ap.add_argument("--mask-scale-y", type=float, default=0.90)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--try-jit", action="store_true")
    ap.add_argument("--queue-size", type=int, default=2)
    ap.add_argument("--drop-policy", choices=["drop_oldest", "drop_new"], default="drop_oldest")
    args = ap.parse_args()

    rt_cfg = RTConfig(
        src_img_path=args.src,
        input_uri=args.input,
        out_path=args.out,
        device=args.device,
        ckpt=args.ckpt,
        id_enc_ckpt=args.id_enc_ckpt,
        detect_every=args.detect_every,
        smooth_alpha_prev=args.smooth_alpha,
        temporal_alpha_cur=args.temporal_alpha,
        blend="feather",
        feather_ksize=args.feather_ksize,
        color_match=bool(args.color_match),
        mask_scale_x=args.mask_scale_x,
        mask_scale_y=args.mask_scale_y,
        cpu_threads=args.cpu_threads,
        try_jit=bool(args.try_jit),
        queue_size=args.queue_size,
        drop_policy=args.drop_policy,
    )

    app = RealTimeFaceSwapper(rt_cfg)
    app.run()


if __name__ == "__main__":
    main()