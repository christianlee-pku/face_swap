import os
import random
from typing import Dict
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

class FaceSwapPairs(Dataset):
    """
    Read pair CSV: columns = [src_path, tgt_path, same_identity]
    Assumes aligned images (256x256) and masks in data/masks_256 mirroring identity/filename.
    """
    def __init__(self, pairs_csv: str, masks_root="data/masks_256", img_size=256, aug=True):
        self.df = pd.read_csv(pairs_csv)
        self.masks_root = masks_root
        self.img_size = img_size
        self.aug = aug
        self.to_tensor = T.Compose([
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)  # -> [-1,1]
        ])
        self.color_aug = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)

    def _mask_path_for(self, aligned_path: str) -> str:
        # aligned_path: data/aligned_256/<id>/<name>.png
        parts = aligned_path.split(os.sep)
        parts[1] = "masks_256"  # replace aligned_256
        return os.sep.join(parts)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        src_p = row["src_path"]; tgt_p = row["tgt_path"]
        same = int(row["same_identity"]) == 1
        src = Image.open(src_p).convert("RGB")
        tgt = Image.open(tgt_p).convert("RGB")
        if self.aug and random.random() < 0.5:
            src = self.color_aug(src); tgt = self.color_aug(tgt)
        src_t = self.to_tensor(src); tgt_t = self.to_tensor(tgt)
        mask_p = self._mask_path_for(tgt_p)
        mask = Image.open(mask_p).convert("L").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        return {
            "src": src_t, "tgt": tgt_t, "tgt_mask": mask,
            "same": torch.tensor(1 if same else 0, dtype=torch.long),
            "src_path": src_p, "tgt_path": tgt_p
        }
