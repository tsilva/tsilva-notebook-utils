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
