import torch
from huggingface_hub import HfApi, create_repo, login, upload_file, whoami
from huggingface_hub.utils import RepositoryNotFoundError

from .torch import get_gpu_stats


def huggingface_login(token: str):
    login(token=token)
    return whoami()

def print_trainer_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    for key, value in get_gpu_stats().items(): print(f"{key}: {value:.4f}")

def hf_create_or_get_repo(repo_name: str, private: bool = True) -> str:
    api = HfApi()
    
    try:
        # Check if repo exists
        _ = api.repo_info(repo_name)
        print(f"Repository '{repo_name}' already exists.")
        return f"https://huggingface.co/{repo_name}"
    except RepositoryNotFoundError:
        # Create if not exists
        repo_url = create_repo(repo_name, private=private)
        print(f"Repository '{repo_name}' created.")
        return repo_url

def hf_push_model_to_hub(repo_id: str, model, model_file_name="model.pt", private: bool = True):
    hf_create_or_get_repo(repo_id, private=private)

    torch.save(model.state_dict(), model_file_name)

    upload_file(
        path_or_fileobj=model_file_name,
        path_in_repo=model_file_name,
        repo_id=repo_id,
        repo_type="model"
    )

