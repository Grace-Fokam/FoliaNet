"""Loads training images from one folder per class.

e.g. data/crop_images/Corn_(maize)___Common_rust_/ and
data/crop_images/Corn_(maize)___healthy/

`crops` filters folders by name (like "Corn" or "Wheat")."""

from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_transforms(img_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CropDiseaseDataset(Dataset):
    def __init__(self, root: str, crops: List[str], transform=None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Can't find '{root}'. See data/README.md to set up a dataset."
            )
        keywords = [c.lower() for c in crops]
        class_dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and any(k in d.name.lower() for k in keywords)
        )
        if not class_dirs:
            raise RuntimeError(
                f"No folders in '{root}' match crops={crops}."
            )
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        for d in class_dirs:
            idx = self.class_to_idx[d.name]
            for f in d.rglob("*"):
                if f.suffix.lower() in _EXTS:
                    self.samples.append((f, idx))
        # Same order every time so the train/val split matches.
        self.samples.sort(key=lambda s: str(s[0]))
        if not self.samples:
            raise RuntimeError(f"No images found in the class folders under '{root}'.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
