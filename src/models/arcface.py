try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


try:
    from facenet_pytorch import InceptionResnetV1  # type: ignore
except Exception:  # pragma: no cover
    InceptionResnetV1 = None


class ArcFaceEmbedder(nn.Module if nn else object):  # type: ignore[misc]
    """ArcFace-like embedder using facenet_pytorch; falls back to pooling if weights unavailable."""

    def __init__(self, pretrained: bool = True):
        if nn:
            super().__init__()
            self.valid = False
            if InceptionResnetV1:
                try:
                    self.model = InceptionResnetV1(pretrained="vggface2" if pretrained else None).eval()
                    self.valid = True
                except Exception:
                    self.model = nn.AdaptiveAvgPool2d((1, 1))
            else:
                self.model = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):  # type: ignore[override]
        if torch is None:
            return x
        if getattr(self, "valid", False):
            with torch.no_grad():
                return self.model(x)
        pooled = self.model(x)
        return pooled.view(pooled.size(0), -1)
