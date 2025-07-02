import os
import platform
import subprocess




def login(*args, **kwargs):
    import wandb

    return wandb.login(*args, **kwargs)


def render_run_iframe():
    import wandb
    from IPython.display import HTML, display

    run_url = wandb.run.get_url()
    iframe_code = f"""
    <iframe src="{run_url}" width="100%" height="1200px"></iframe>
    """

    display(HTML(iframe_code))


def runtime_metadata():
    import torch

    # --- Runtime metadata -----------------------------------------------------
    def _get_git_commit() -> str:
        """Return the short SHA if this is a Git repo, else 'unknown'."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"

    return {
        "git_commit": _get_git_commit(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda or "cpu",
        "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "python_version": platform.python_version(),
        "run_host": os.uname().nodename,
    }
