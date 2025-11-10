import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetV2Perceptual(nn.Module):
    """
    Lightweight perceptual loss using ImageNet-pretrained MobileNetV2 features.
    - Used ONLY during training to keep pose/expression/structure.
    - MobileNetV2 is frozen (no grads), so it does not add trainable params.

    Args:
        layers: indices inside mobilenet_v2.features to tap for feature matching.
                Default taps capture low/mid/high levels.
        reduction: 'l1' or 'l2' distance between features.
        normalize: if True, feature maps are L2-normalized per-channel before the loss.

    Inputs:
        x, y: images in [-1, 1], shape Bx3xHxW

    Output:
        scalar loss (torch.Tensor)
    """
    def __init__(self, layers=(2, 4, 7, 14), reduction: str = "l1", normalize: bool = False):
        super().__init__()
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.layers = set(layers)
        self.reduction = reduction
        self.normalize = normalize
        self._eps = 1e-8

    @torch.no_grad()
    def _forward_feats(self, img: torch.Tensor):
        # convert from [-1,1] -> [0,1] because the backbone was trained on [0,1]/ImageNet stats
        x = (img * 0.5) + 0.5
        feats = []
        h = x
        for i, m in enumerate(self.backbone):
            h = m(h)
            if i in self.layers:
                if self.normalize:
                    # L2-normalize per-channel to reduce scale bias across layers
                    n = torch.sqrt((h * h).mean(dim=(2,3), keepdim=True) + self._eps)
                    feats.append(h / n)
                else:
                    feats.append(h)
        return feats

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        fx = self._forward_feats(x)
        fy = self._forward_feats(y)
        loss = 0.0
        for a, b in zip(fx, fy):
            if self.reduction == "l2":
                loss = loss + F.mse_loss(a, b)
            else:
                loss = loss + F.l1_loss(a, b)
        return loss

def cosine_id_loss(embed_out: torch.Tensor, embed_src: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine identity loss: 1 - cos(emb_out, emb_src). Lower is better.
    Embeddings are L2-normalized inside.
    """
    eo = embed_out / (embed_out.norm(dim=1, keepdim=True) + eps)
    es = embed_src / (embed_src.norm(dim=1, keepdim=True) + eps)
    cos = (eo * es).sum(dim=1)
    return 1.0 - cos.mean()