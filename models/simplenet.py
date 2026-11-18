import torch, torch.nn as nn, torch.nn.functional as F
from .backbone import Backbone

class SimpleNetPP(nn.Module):
    def __init__(self, backbone="repvit", dim=1536, sigma=0.015, th=0.5):
        super().__init__()
        self.backbone = Backbone(backbone)
        self.dim, self.sigma, self.th = dim, sigma, th
        self.adaptor = nn.Linear(dim, dim)                       # feature adapter g_phi
        self.disc = nn.Sequential(nn.Linear(dim, dim // 2), nn.BatchNorm1d(dim // 2),
                                  nn.LeakyReLU(0.2), nn.Linear(dim // 2, 1, bias=False))

    @torch.no_grad()
    def extract(self, x):
        feats = self.backbone(x)
        size = feats[0].shape[-2:]
        out = []
        for f in feats:
            f = F.avg_pool2d(f, 3, 1, 1)                          # local neighbourhood aggregation
            f = F.interpolate(f, size=size, mode="bilinear", align_corners=False)
            B, C, H, W = f.shape
            f = f.permute(0, 2, 3, 1).reshape(-1, 1, C)
            out.append(F.adaptive_avg_pool1d(f, self.dim).squeeze(1))
        z = torch.stack(out, 1)                                   # (BHW, L, dim)
        z = F.adaptive_avg_pool1d(z.reshape(z.shape[0], 1, -1), self.dim).squeeze(1)
        return z, (B, H, W)

    def train_loss(self, x):
        with torch.no_grad():
            z, _ = self.extract(x)
        z = self.adaptor(z)
        z_a = z + torch.randn_like(z) * self.sigma                # synthetic anomalies
        s_n, s_a = self.disc(z), self.disc(z_a)
        return F.relu(self.th - s_n).mean() + F.relu(s_a + self.th).mean()  # truncated L1

    @torch.no_grad()
    def predict(self, x):
        z, (B, H, W) = self.extract(x)
        s = -self.disc(self.adaptor(z)).reshape(B, 1, H, W)       # higher = more anomalous
        s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)
        s = gaussian_blur(s, sigma=4)
        return s.squeeze(1), s.flatten(1).max(1).values

def gaussian_blur(x, sigma=4):
    k = int(4 * sigma) * 2 + 1
    ax = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = (g / g.sum()).to(x.dtype)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
