from typing import Callable, Any

def huggingface_login(token: str):
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


def process_images(images, mode="color", quantize_colors=None, scale=1.0, crop_paddings=None, noise_factor=0.0, return_type="pt"):
    import numpy as np
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
            import torch
            image_ts = to_tensor(image).to(dtype=torch.float16)
            processed.append(image_ts)
        elif return_type == "np":
            image_np = np.array(image).astype(np.float16) / 255.0
            processed.append(image_np)

    return processed
