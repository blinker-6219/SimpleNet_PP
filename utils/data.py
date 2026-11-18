import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
CATS = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill",
        "screw","tile","toothbrush","transistor","wood","zipper"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class MVTec(Dataset):
    def __init__(self, root, cat, split, resize=329, size=288):
        self.paths, self.masks = [], []
        base = os.path.join(root, cat)
        if split == "train":
            self.paths = sorted(glob.glob(f"{base}/train/good/*.png"))
            self.masks = [None] * len(self.paths)
        else:
            for d in sorted(os.listdir(f"{base}/test")):
                for p in sorted(glob.glob(f"{base}/test/{d}/*.png")):
                    self.paths.append(p)
                    self.masks.append(None if d == "good" else
                        os.path.join(base, "ground_truth", d, os.path.basename(p).replace(".png", "_mask.png")))
        self.size = size
        self.tf = T.Compose([T.Resize(resize), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
        self.mtf = T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
                              T.CenterCrop(size), T.ToTensor()])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = self.tf(Image.open(self.paths[i]).convert("RGB"))
        if self.masks[i] is None:
            m, y = torch.zeros(1, self.size, self.size), 0
        else:
            m, y = (self.mtf(Image.open(self.masks[i]).convert("L")) > 0.5).float(), 1
        return img, m, y
