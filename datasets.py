"""
datasets.py
Unpaired CT (domain A) <-> MRI (domain B) dataset loader.

Expected directory layout (matches the Kaggle "CT-to-MRI-CGAN" dataset,
darren2020/ct-to-mri-cgan, which contains 4972 unpaired grayscale 512x512
slices split into trainA/trainB/testA/testB):

    data/
        trainA/   *.png|jpg   (CT)
        trainB/   *.png|jpg   (MRI)
        testA/    *.png|jpg   (CT)
        testB/    *.png|jpg   (MRI)
"""

import os
import random
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def build_transforms(img_size, train=True):
    ops = [T.Resize((img_size, img_size), Image.BICUBIC)]
    if train:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(5)], p=0.3),
        ]
    ops += [
        T.ToTensor(),                # [0,1]
        T.Normalize(mean=[0.5], std=[0.5]),  # -> [-1,1], matches Tanh generator output
    ]
    return T.Compose(ops)


class UnpairedCTMRIDataset(Dataset):
    """Returns unpaired (CT, MRI) samples. Since CycleGAN needs no pairing,
    domain B index is drawn independently (randomly permuted each epoch)."""

    def __init__(self, dir_a, dir_b, img_size=256, train=True, channels=1):
        self.files_a = _list_images(dir_a)
        self.files_b = _list_images(dir_b)
        if len(self.files_a) == 0 or len(self.files_b) == 0:
            raise RuntimeError(
                f"No images found. Checked:\n  A: {dir_a} ({len(self.files_a)} files)\n"
                f"  B: {dir_b} ({len(self.files_b)} files)\n"
                f"Download the dataset from "
                f"https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan and place "
                f"trainA/trainB/testA/testB under ./data/"
            )
        self.transform = build_transforms(img_size, train)
        self.train = train
        self.channels = channels
        self.mode = "L" if channels == 1 else "RGB"

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))

    def _load(self, path):
        img = Image.open(path).convert(self.mode)
        return self.transform(img)

    def __getitem__(self, idx):
        a_path = self.files_a[idx % len(self.files_a)]
        if self.train:
            b_path = self.files_b[random.randint(0, len(self.files_b) - 1)]
        else:
            b_path = self.files_b[idx % len(self.files_b)]
        return {
            "A": self._load(a_path),
            "B": self._load(b_path),
            "A_path": a_path,
            "B_path": b_path,
        }


class ReplayBuffer:
    """Historical buffer of generated (fake) images, as used in the original
    CycleGAN implementation, to reduce oscillation during adversarial training."""

    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, batch):
        out = []
        for element in batch.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                out.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    out.append(self.data[idx].clone())
                    self.data[idx] = element
                else:
                    out.append(element)
        return torch.cat(out)
