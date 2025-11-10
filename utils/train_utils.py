from __future__ import annotations
import random
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset_pairs import FaceSwapPairs

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def collate_dict(batch: list[Dict[str, Any]]):
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out

def filter_pairs_csv(in_csv: str, same_only: bool, out_csv: Optional[str] = None) -> str:
    if not same_only:
        return in_csv
    df = pd.read_csv(in_csv)
    df = df[df["same_identity"] == 1].reset_index(drop=True)
    if out_csv is None:
        p = Path(in_csv)
        out_csv = str(p.with_name(p.stem + "_same.csv"))
    df.to_csv(out_csv, index=False)
    return out_csv

def make_loader(pairs_csv: str, batch_size: int, aug: bool, workers: int):
    ds = FaceSwapPairs(pairs_csv, aug=aug)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=aug, num_workers=workers,
        pin_memory=torch.cuda.is_available(), collate_fn=collate_dict
    )
    return ds, dl

def current_lr(optim) -> float:
    for g in optim.param_groups:
        return float(g.get("lr", 0.0))
    return 0.0

def safe_item(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")
