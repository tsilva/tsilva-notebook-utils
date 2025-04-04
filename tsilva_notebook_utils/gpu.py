import torch

def get_current_device():
    if not torch.cuda.is_available(): return torch.device("cpu")

    # Get the current device (default is GPU 0 if not specified)
    device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    # Memory stats
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # Convert to GB
    allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
    cached_memory = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB
    free_memory = total_memory - allocated_memory

    print(f"Total GPU Memory: {total_memory:.2f} GB")
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Cached Memory: {cached_memory:.2f} GB")
    print(f"Free Memory (approx): {free_memory:.2f} GB")

def print_gpu_utilization():
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
