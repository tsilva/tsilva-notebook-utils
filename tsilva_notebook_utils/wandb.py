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
