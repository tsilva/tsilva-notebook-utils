import json
import random
from typing import Any, Callable

try:
    from datasets import Dataset
except Exception:
    Dataset = object  # type: ignore


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

        def _process(_x):
            import torch
            if random.random() < self.prob:
                if random.random() < self.noise_prob:
                    noise = torch.randn_like(_x) * self.noise_std
                    _x = torch.clamp(_x + noise, 0.0, 1.0)

                if random.random() < self.flip_prob:
                    if random.random() < self.hflip_prob:
                        _x = torch.flip(_x, dims=[2])  # Horizontal flip
                    if random.random() < self.vflip_prob:
                        _x = torch.flip(_x, dims=[1])  # Vertical flip
            return _x
        
        x = [_process(_x) for _x in x]
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


def process_images(images, mode="color", quantize_colors=None, scale=1.0, crop_paddings=None, noise_factor=0.0, return_type="pt"):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    processed = []
    for image in images:
        # Crop if needed
        if crop_paddings:
            width, height = image.size
            left = crop_paddings[3]
            top = crop_paddings[0]
            right = width - crop_paddings[1]
            bottom = height - crop_paddings[2]
            image = image.crop((left, top, right, bottom))


        # Apply color quantization (after conversion & crop, before tensor)
        if quantize_colors:
            image = image.quantize(colors=quantize_colors)

        # Resize
        if scale != 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)

        # Apply noise (directly on PIL image)
        if noise_factor:
            np_image = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, noise_factor, np_image.shape)
            noisy_np = np.clip(np_image + noise, 0.0, 1.0) * 255.0
            image = Image.fromarray(noisy_np.astype(np.uint8), mode=image.mode)

        # Convert mode
        if mode == "color":
            image = image.convert("RGB")
        elif mode == "grayscale":
            image = image.convert("L")
        elif mode == "black_and_white":
            image = image.convert("1")
        else:
            raise ValueError(f"Unsupported color mode: {mode}")

        if return_type == "pt":
            image_ts = to_tensor(image).to(dtype=torch.float16)
            processed.append(image_ts)
        elif return_type == "np":
            image_np = np.array(image).astype(np.float16) / 255.0
            processed.append(image_np)

    return processed


def dedupe_dataset(dataset, feature_key: str, hash_func: Callable[[Any], bytes] = None):
    seen_hashes = set()

    def default_hash_func(value):
        if hasattr(value, "tobytes"):
            return value.tobytes()  # e.g., NumPy arrays, PIL images
        elif isinstance(value, str):
            return value.encode("utf-8")
        else:
            return json.dumps(value, sort_keys=True).encode("utf-8")

    _hash_func = hash_func or default_hash_func

    def _filter(row):
        value = row[feature_key]
        hashed = _hash_func(value)
        if hashed in seen_hashes:
            return False
        seen_hashes.add(hashed)
        return True

    return dataset.filter(_filter)
