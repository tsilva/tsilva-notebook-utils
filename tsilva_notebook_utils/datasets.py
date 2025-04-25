import torch
import random
from datasets import Dataset

class AugmentedImageDataset(Dataset):
    def __init__(
        self, 
        base_dataset, 
        prob=0.0, 
        noise_prob=0.0,
        noise_std=0.0, 
        flip_prob=0.0,
        hflip_prob=0.0, 
        vflip_prob=0.0
    ):
        # Base dataset to wrap and augment on-the-fly
        self.base_dataset = base_dataset

        # Probability settings
        self.prob = prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        x = item["x"].clone()
        y = item["y"]

        if random.random() < self.prob:
            if random.random() < self.noise_prob:
                noise = torch.randn_like(x) * self.noise_std
                x = torch.clamp(x + noise, 0.0, 1.0)

            if random.random() < self.flip_prob:
                if random.random() < self.hflip_prob:
                    x = torch.flip(x, dims=[2])  # Horizontal flip
                if random.random() < self.vflip_prob:
                    x = torch.flip(x, dims=[1])  # Vertical flip

        return {"x": x, "y": y}

    def __len__(self):
        return len(self.base_dataset)


class ShiftedDataset(Dataset):
    def __init__(self, base_dataset, shift=1):
        if shift < 1: raise ValueError("Shift must be >= 1")
        self.base_dataset = base_dataset
        self.shift = shift

    def __len__(self):
        return len(self.base_dataset) - self.shift

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self): raise IndexError("Index out of range")
        x = self.base_dataset[idx]
        y = self.base_dataset[idx + self.shift]
        return {"x": x.get("x"), "y": y.get("y")}

    def __len__(self):
        return len(self.base_dataset)
