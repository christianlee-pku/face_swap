import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import torch
    from torchvision import transforms as T
except Exception:  # pragma: no cover
    torch = None
    T = None

from .manifest import DatasetManifest


class LFWDataset:
    """LFW dataset wrapper using a manifest (paths, splits, optional transforms)."""

    def __init__(
        self,
        root: str,
        split: str,
        manifest: str,
        transform=None,
        to_tensor: bool = True,
        sample_ratio: float = 1.0,
        sample_ratio_overrides: Optional[Dict[str, float]] = None,
        sample_seed: int = 42,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.to_tensor = to_tensor and T is not None
        self.tensor_tf = T.ToTensor() if self.to_tensor and T else None
        self.manifest_path = Path(manifest)
        self.samples: List[Dict[str, Any]] = []
        ratio_overrides = sample_ratio_overrides or {}
        ratio = ratio_overrides.get(split, sample_ratio)
        ratio = max(0.0, min(1.0, ratio))
        if self.manifest_path.exists():
            manifest_data = DatasetManifest.load(self.manifest_path)
            ids = manifest_data.splits.get(split, [])
            items_by_id = {item["id"]: item for item in manifest_data.items}
            picked = [items_by_id[i] for i in ids if i in items_by_id]
            if ratio < 1.0 and len(picked) > 0:
                rng = random.Random(sample_seed)
                k = max(1, int(len(picked) * ratio))
                picked = rng.sample(picked, k=k)
            self.samples = picked
        else:
            # Fallback empty list if manifest missing
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Any:
        if Image and path.exists():
            return Image.open(path).convert("RGB")
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.samples[idx]) if self.samples else {"id": idx, "path": "", "label": ""}
        img = None
        if "path" in sample:
            img_path = self.root / sample["path"]
            img = self._load_image(img_path)
            sample["image"] = img
            sample["image_path"] = str(img_path)
        if self.transform:
            sample = self.transform(sample)
        if self.tensor_tf and sample.get("image") is not None:
            sample["image_tensor"] = self.tensor_tf(sample["image"])
            # duplicate as target for now
            sample["target_tensor"] = sample["image_tensor"]
        # Drop raw PIL image to avoid DataLoader collate issues.
        sample.pop("image", None)
        return sample
