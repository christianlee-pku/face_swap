import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
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

class FiLM(nn.Module):
    """Channel-wise FiLM: (1+gamma)*x + beta, gamma/beta from id embedding"""
    def __init__(self, id_dim, num_ch):
        super().__init__()
        self.fc = nn.Linear(id_dim, num_ch * 2)
    def forward(self, x, id_emb):
        B, C, H, W = x.shape
        gb = self.fc(id_emb)      # Bx(2C)
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)
        return x * (1.0 + gamma) + beta

class MobileSwapLite(nn.Module):
    """
    Input:  tgt: Bx3x256x256 in [-1,1], id_emb: Bx128
    Output: out: Bx3x256x256 in [-1,1]
    """
    def __init__(self, base=24, id_dim=128):
        super().__init__()
        c1 = base         # 24
        c2 = base * 2     # 48
        c3 = base * 4     # 96
        # Encoder
        self.enc1 = DSConv(3,  c1, stride=1)  # 256
        self.enc2 = DSConv(c1, c2, stride=2)  # 128
        self.enc3 = DSConv(c2, c2, stride=1)  # 128
        self.enc4 = DSConv(c2, c3, stride=2)  # 64
        self.enc5 = DSConv(c3, c3, stride=1)  # 64
        self.enc6 = DSConv(c3, c3, stride=2)  # 32 (bottleneck)
        # FiLM
        self.film_bot = FiLM(id_dim, c3)
        self.film_up2 = FiLM(id_dim, c3)
        self.film_up1 = FiLM(id_dim, c2)
        self.film_up0 = FiLM(id_dim, c1)
        # Decoder
        self.up2_conv = DSConv(c3 + c3, c3, stride=1)  # 32->64 concat enc5
        self.up1_conv = DSConv(c3 + c2, c2, stride=1)  # 64->128 concat enc3
        self.up0_conv = DSConv(c2 + c1, c1, stride=1)  # 128->256 concat enc1
        self.to_rgb = nn.Conv2d(c1, 3, 1)

    def forward(self, tgt, id_emb):
        f1 = self.enc1(tgt)         # 256,c1
        x  = self.enc2(f1)          # 128,c2
        f2 = self.enc3(x)           # 128,c2
        x  = self.enc4(f2)          # 64,c3
        f3 = self.enc5(x)           # 64,c3
        x  = self.enc6(f3)          # 32,c3
        x  = self.film_bot(x, id_emb)
        # 32->64
        x  = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x  = torch.cat([x, f3], dim=1)
        x  = self.up2_conv(x); x = self.film_up2(x, id_emb)
        # 64->128
        x  = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x  = torch.cat([x, f2], dim=1)
        x  = self.up1_conv(x); x = self.film_up1(x, id_emb)
        # 128->256
        x  = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x  = torch.cat([x, f1], dim=1)
        x  = self.up0_conv(x); x = self.film_up0(x, id_emb)
        out = torch.tanh(self.to_rgb(x))
        return out