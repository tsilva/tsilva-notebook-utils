def init_with_defaults(config):
    import os
    import wandb
    from tsilva_notebook_utils.wandb import login

    NOTEBOOK_ID = os.getenv("NOTEBOOK_ID"); assert NOTEBOOK_ID is not None, "NOTEBOOK_ID environment variable is not set"
    WANDB_API_KEY = os.getenv("WANDB_API_KEY"); assert WANDB_API_KEY is not None, "WANDB_API_KEY environment variable is not set"
    notebook_id = config.get("notebook_id"); assert notebook_id is not None, "notebook_id key is not set in config"

    # Login to W&B
    login(WANDB_API_KEY, notebook_id)

    # Initialize a W&B run for training
    return wandb.init(project=notebook_id, config=config)

def login(api_key: str, project_id: str) -> str:
    import wandb

    # Login to W&B
    wandb.login(key=api_key)

    # Initialize a W&B run for training
    run = wandb.init(
        project=project_id,
        reinit=True
    )

    # Return the run URL
    run_url = run.get_url()
    return run_url

def render_run_iframe():
    import wandb
    from IPython.display import HTML
    from IPython.display import display

    run_url = wandb.run.get_url()
    iframe_code = f"""
    <iframe src="{run_url}" width="100%" height="1200px"></iframe>
    """

    display(HTML(iframe_code))
