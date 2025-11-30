try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional torch
    torch = None
    nn = None


def _conv_block(in_ch: int, out_ch: int) -> "nn.Module":  # type: ignore[name-defined]
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetFaceSwap(nn.Module if nn else object):  # type: ignore[misc]
    """UNet-style model aimed at preserving identity/expression/skin tone."""

    def __init__(self, channels: int = 64):
        if nn:
            super().__init__()
            self.channels = channels
            # accept concatenated source+target: 3+3 channels
            self.down1 = _conv_block(6, channels)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = _conv_block(channels, channels * 2)
            self.pool2 = nn.MaxPool2d(2)

            self.bottleneck = _conv_block(channels * 2, channels * 4)

            self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)
            self.dec2 = _conv_block(channels * 4, channels * 2)
            self.up1 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
            self.dec1 = _conv_block(channels * 2, channels)
            self.out_conv = nn.Conv2d(channels, 3, kernel_size=1)
        else:  # pragma: no cover
            self.channels = channels

    def forward(self, source, target=None):  # type: ignore[override]
        if nn is None:
            out = source if target is None else target
            return {"output": out}
        tgt = target if target is not None else source
        x = torch.cat([source, tgt], dim=1)
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        bottleneck = self.bottleneck(p2)

        u2 = self.up2(bottleneck)
        concat2 = torch.cat([u2, d2], dim=1)
        dec2 = self.dec2(concat2)
        u1 = self.up1(dec2)
        concat1 = torch.cat([u1, d1], dim=1)
        dec1 = self.dec1(concat1)
        out = self.out_conv(dec1)
        return {"output": out}
