import random
from typing import Any, Dict

try:
    import torchvision.transforms as T
    from PIL import Image
except Exception:  # pragma: no cover
    T = None
    Image = None


class LightAugmentation:
    """Deterministic light augmentations (flip/jitter/blur) if torchvision available."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        if T:
            self.transform = T.Compose(
                [
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                    T.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:  # pragma: no cover
            self.transform = None

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = dict(sample)
        if T and Image and "image" in sample and isinstance(sample["image"], Image.Image):
            random.seed(self.seed)
            sample["image"] = self.transform(sample["image"]) if self.transform else sample["image"]
        sample["aug_seed"] = self.seed
        return sample
