def render_run_iframe():
    import wandb
    from IPython.display import HTML
    from IPython.display import display

    run_url = wandb.run.get_url()
    iframe_code = f"""
    <iframe src="{run_url}" width="100%" height="1200px"></iframe>
    """

    display(HTML(iframe_code))
