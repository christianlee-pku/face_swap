import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    """Depthwise Separable: DW(3x3)+BN+ReLU6 -> PW(1x1)+BN+ReLU6"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.dw_bn(x); x = self.act(x)
        x = self.pw(x); x = self.pw_bn(x); x = self.act(x)
        return x

class IDEncoderMobile(nn.Module):
    """
    Mobile-friendly ID encoder.
    Input:  Bx3x256x256 in [-1,1]
    Output: Bx128 L2-normalized embedding
    """
    def __init__(self, width=1.0, emb_dim=128):
        super().__init__()
        c1 = int(16 * width)
        c2 = int(32 * width)
        c3 = int(64 * width)
        c4 = int(96 * width)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU6(inplace=True)
        )
        self.b1 = DSConv(c1, c2, stride=2)   # 256->128
        self.b2 = DSConv(c2, c3, stride=2)   # 128->64
        self.b3 = DSConv(c3, c4, stride=2)   # 64->32
        self.b4 = DSConv(c4, c4, stride=1)   # refine
        self.head = nn.Linear(c4, emb_dim)

    def forward(self, x):
        x = (x * 0.5) + 0.5             # to [0,1]
        x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # BxC
        x = self.head(x)                 # Bxemb
        x = F.normalize(x, p=2, dim=1)   # L2 norm
        return x
