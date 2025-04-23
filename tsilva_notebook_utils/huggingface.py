
from typing import Callable, Any

def login(token: str):
    from huggingface_hub import login, whoami
    login(token=token)
    return whoami()

def print_trainer_summary(result):
    from .gpu import get_gpu_stats
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    for key, value in get_gpu_stats().items(): print(f"{key}: {value:.4f}")

def dedupe_dataset(dataset, feature_key: str, hash_func: Callable[[Any], bytes] = None):
    seen_hashes = set()

    def default_hash_func(value):
        if hasattr(value, "tobytes"):
            return value.tobytes()  # e.g., NumPy arrays, PIL images
        elif isinstance(value, str):
            return value.encode("utf-8")
        else:
            import json
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


def process_images(images, mode="color", quantize_colors=None, scale=1.0, crop_paddings=None, noise_factor=0.0):
    import torch
    from torchvision.transforms.functional import to_tensor, resize

    processed = []
    for image in images:
        # Apply color quantization (after conversion & crop, before tensor)
        if quantize_colors:
            image = image.quantize(colors=quantize_colors)

        # Convert mode
        if mode == "color":
            image = image.convert("RGB")
        elif mode == "grayscale":
            image = image.convert("L")
        elif mode == "black_and_white":
            image = image.convert("1")
        else:
            raise ValueError(f"Unsupported color mode: {mode}")

        # Crop if needed
        if crop_paddings:
            width, height = image.size
            left = crop_paddings[3]
            top = crop_paddings[0]
            right = width - crop_paddings[1]
            bottom = height - crop_paddings[2]
            image = image.crop((left, top, right, bottom))

        # Convert to tensor
        image_ts = to_tensor(image)

        # Resize
        if scale != 1.0:
            _, h, w = image_ts.shape
            new_h, new_w = int(h * scale), int(w * scale)
            image_ts = resize(image_ts, [new_h, new_w])

        # Apply noise
        if noise_factor:
            noise = torch.randn_like(image_ts) * noise_factor
            image_ts = torch.clamp(image_ts + noise, 0.0, 1.0)

        # Convert to float16
        image_ts = image_ts.to(torch.float16)
        processed.append(image_ts)

    return processed
