from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from models.id_encoder_mobile import IDEncoderMobile
from models.generator_mobileswaplite import MobileSwapLite
from utils.face_align_infer import (
    mtcnn_detector, align_face_256, apply_M, warp_by_M, build_face_mask_256
)
from utils.blend import blend_feathered, poisson_blend, match_lightness_hsv

@dataclass
class InferenceConfig:
    ckpt_path: str
    id_enc_ckpt_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    blend_mode: str = "feather"   # "feather" | "poisson"
    feather_ksize: int = 25
    color_match: bool = False     # HSV lightness match
    src_index: Optional[int] = None
    tgt_index: Optional[int] = None
    mask_scale_x: float = 1.0
    mask_scale_y: float = 1.0

class FaceSwapInference:
    """
    End-to-end face swap engine (image-level).
    """
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.id_enc = IDEncoderMobile().to(self.device).eval()
        if cfg.id_enc_ckpt_path:
            sd = torch.load(cfg.id_enc_ckpt_path, map_location=self.device)
            if isinstance(sd, dict) and "state_dict" in sd:
                self.id_enc.load_state_dict(sd["state_dict"], strict=True)
            else:
                self.id_enc.load_state_dict(sd, strict=True)
        # 2) Generator
        self.gen = MobileSwapLite().to(self.device).eval()
        state = torch.load(cfg.ckpt_path, map_location=self.device)
        self.gen.load_state_dict(state["gen"], strict=True)
        # 3) Detector
        self.det = mtcnn_detector(device=self.device)

    @torch.no_grad()
    def _to_tensor_norm(self, arr_uint8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(arr_uint8).float() / 255.0
        x = x.permute(2,0,1).unsqueeze(0)              # 1x3xHxW
        x = (x - 0.5) / 0.5
        return x.to(self.device)

    @torch.no_grad()
    def swap_images(self, src_img: np.ndarray, tgt_img: np.ndarray) -> Dict:
        # 1) detect+align both
        src_aligned, M_src, _, land_src_aligned, info_src = align_face_256(src_img, self.det, self.cfg.src_index)
        if not info_src["ok"]:
            raise RuntimeError("No face detected in source image")
        tgt_aligned, M_tgt, M_tgt_inv, land_tgt_aligned, info_tgt = align_face_256(tgt_img, self.det, self.cfg.tgt_index)
        if not info_tgt["ok"]:
            raise RuntimeError("No face detected in target image")

        # 2) mask in aligned space
        mask_aligned = build_face_mask_256(
            land_tgt_aligned,
            scale_x=self.cfg.mask_scale_x,
            scale_y=self.cfg.mask_scale_y
        )

        # 3) encode source ID from aligned source
        src_tensor = self._to_tensor_norm(src_aligned)
        id_emb = self.id_enc(src_tensor)   # 1x128

        # 4) generate aligned output for target
        tgt_tensor = self._to_tensor_norm(tgt_aligned)
        out_tensor = self.gen(tgt_tensor, id_emb)      # 1x3x256x256 in [-1,1]
        out_aligned = ((out_tensor.squeeze(0).permute(1,2,0).clamp(-1,1).cpu().numpy() * 0.5 + 0.5) * 255.0).astype(np.uint8)
        # out_aligned = tgt_aligned.copy()

        # 5) optional color/lightness matching
        paste_aligned = out_aligned
        if self.cfg.color_match:
            paste_aligned = match_lightness_hsv(out_aligned, tgt_aligned)

        # 6) inverse warp paste and mask back to target original
        import cv2
        paste = cv2.warpAffine(paste_aligned, M_tgt_inv, (tgt_img.shape[1], tgt_img.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine((mask_aligned*255).astype(np.uint8), M_tgt_inv,
                              (tgt_img.shape[1], tgt_img.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) / 255.0
        mask = np.clip(mask, 0.0, 1.0)

        # 7) blend
        if self.cfg.blend_mode == "poisson":
            center_pt = np.array([[128.0, 140.0]], dtype=np.float32)  # heuristic
            center = apply_M(center_pt, M_tgt_inv)[0].astype(int)
            composite = poisson_blend(tgt_img, paste, mask, center=tuple(center))
        else:
            composite = blend_feathered(tgt_img, paste, mask, ksize=self.cfg.feather_ksize)

        return {
            "composite": composite,
            "debug": {
                "src_det": info_src, "tgt_det": info_tgt,
                "blend_mode": self.cfg.blend_mode, "color_match": self.cfg.color_match,
                "mask_scale": (self.cfg.mask_scale_x, self.cfg.mask_scale_y)
            }
        }
