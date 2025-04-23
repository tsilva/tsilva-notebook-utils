
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
