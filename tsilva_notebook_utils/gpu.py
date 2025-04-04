import torch

def get_current_device():
    return torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

def get_gpu_stats():
    assert torch.cuda.is_available(), "CUDA is not available. Running on CPU."

    device = get_current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
    cached_memory = torch.cuda.memory_reserved(device) / 1024**3
    free_memory = total_memory - allocated_memory

    return {
        "total_memory_gb": total_memory,
        "allocated_memory_gb": allocated_memory,
        "cached_memory_gb": cached_memory,
        "free_memory_gb": free_memory
    }
