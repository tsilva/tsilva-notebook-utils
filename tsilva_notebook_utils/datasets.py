import torch
import random
from datasets import IterableDataset
from itertools import tee, islice

class AugmentedImageDataset(torch.utils.data.Dataset):
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

        # Probability of applying any augmentation
        self.prob = prob

        # Probability of adding Gaussian noise to the image
        self.noise_prob = noise_prob

        # Standard deviation of the Gaussian noise
        self.noise_std = noise_std

        # Probability of applying any kind of flipping (horizontal or vertical)
        self.flip_prob = flip_prob

        # Probability of applying horizontal flip if flipping is triggered
        self.hflip_prob = hflip_prob

        # Probability of applying vertical flip if flipping is triggered
        self.vflip_prob = vflip_prob

    def __getitem__(self, idx):
        # Fetch the item from the base dataset
        item = self.base_dataset[idx]

        # Clone the image tensor to avoid modifying the original data
        x = item["x"].clone()
        y = item["y"]

        # Decide whether to apply augmentations based on self.prob
        if random.random() < self.prob:
            # Apply Gaussian noise if within noise_prob threshold
            if random.random() < self.noise_prob:
                noise = torch.randn_like(x) * self.noise_std
                x = torch.clamp(x + noise, 0.0, 1.0)  # Clamp to keep pixel values in [0,1]

            # Apply flips if within flip_prob threshold
            if random.random() < self.flip_prob:
                # Horizontal flip
                if random.random() < self.hflip_prob:
                    x = torch.flip(x, dims=[2])  # Flip along width axis (W)

                # Vertical flip
                if random.random() < self.vflip_prob:
                    x = torch.flip(x, dims=[1])  # Flip along height axis (H)

        # Return the potentially augmented item
        return {"x": x, "y": y}

    def __len__(self):
        # Return length of the base dataset
        return len(self.base_dataset)


class ShiftedDataset(IterableDataset):
    def __init__(self, hf_dataset, shift=1):
        if shift < 1:
            raise ValueError("Shift must be >= 1")
        self.dataset = hf_dataset
        self.shift = shift

    def __iter__(self):
        it1, it2 = tee(self.dataset.__iter__(), 2)
        it2 = islice(it2, self.shift, None)

        for x, y in zip(it1, it2):
            yield {
                "x": x.get("x"),
                "y": y.get("y"),
            }

    def __len__(self):
        try:
            return len(self.dataset) - self.shift
        except TypeError:
            raise TypeError("Length is not defined for streaming datasets.")

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        x = self.dataset[idx]
        y = self.dataset[idx + self.shift]
        return {"x": x.get("x"), "y": y.get("y")}

