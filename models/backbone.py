import timm, torch.nn as nn
CFG = {  # timm name, feature-stage indices (stride 8 and 16)
    "repvit":      ("repvit_m0_9", (1, 2)),
    "mobilevit":   ("mobilevit_s", (2, 3)),
    "resnet18":    ("resnet18", (2, 3)),
    "mobilenetv3": ("mobilenetv3_large_100", (2, 3)),
}
class Backbone(nn.Module):
    def __init__(self, key):
        super().__init__()
        name, idx = CFG[key]
        self.net = timm.create_model(name, pretrained=True, features_only=True, out_indices=idx)
        for p in self.net.parameters():
            p.requires_grad = False        # frozen, as in paper
        self.net.eval()
        self.channels = self.net.feature_info.channels()
    def train(self, mode=True):
        return super().train(False)        # keep BN/LN frozen
    def forward(self, x):
        return self.net(x)
