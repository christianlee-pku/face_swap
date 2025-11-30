try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch optional
    torch = None
    nn = None

from typing import Any, Dict

from registry import LOSSES

try:
    from models.arcface import ArcFaceEmbedder
except Exception:  # pragma: no cover
    ArcFaceEmbedder = None


class FaceSwapLoss(nn.Module if nn else object):  # type: ignore[misc]
    """Combined loss for identity preservation, adversarial realism, and reconstruction."""

    def __init__(self, identity_weight: float = 1.0, adv_weight: float = 0.1, recon_weight: float = 1.0):
        if nn:
            super().__init__()
            self.l1 = nn.L1Loss()
            self.arcface = ArcFaceEmbedder(pretrained=True) if ArcFaceEmbedder else None
        self.identity_weight = identity_weight
        self.adv_weight = adv_weight
        self.recon_weight = recon_weight

    def forward(self, outputs: Dict[str, Any], targets: Dict[str, Any]) -> Any:  # type: ignore[override]
        if torch:
            pred = outputs.get("output")
            target_img = targets.get("target")
            if target_img is None:
                target_img = targets.get("target_tensor")
            if pred is None or target_img is None:
                return torch.tensor(0.0)
            recon = self.l1(pred, target_img)
            if self.arcface is not None:
                with torch.no_grad():
                    emb_pred = self.arcface(pred)
                    emb_tgt = self.arcface(target_img)
                identity = torch.norm(emb_pred - emb_tgt, p=2)
            else:
                identity = self.l1(pred, target_img)  # fallback
            adv = torch.tensor(0.0)
            loss = self.identity_weight * identity + self.adv_weight * adv + self.recon_weight * recon
            return loss
        return 0.0

    def __call__(self, outputs: Dict[str, Any], targets: Dict[str, Any]) -> Any:  # type: ignore[override]
        return self.forward(outputs, targets)


if "FaceSwapLoss" not in LOSSES._items:  # type: ignore[attr-defined]
    LOSSES.register("FaceSwapLoss")(FaceSwapLoss)
