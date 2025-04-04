def login(token: str):
    from huggingface_hub import login, whoami
    login(token=token)
    return whoami()

def print_trainer_summary(result):
    from .gpu import print_gpu_utilization
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
