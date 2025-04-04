def login(token: str):
    from huggingface_hub import login, whoami
    login(token=token)
    return whoami()

def print_trainer_summary(result):
    from .gpu import get_gpu_stats
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    for key, value in get_gpu_stats().items(): print(f"{key}: {value:.4f}")
            
