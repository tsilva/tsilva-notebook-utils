def setup_env(config):
    from .torch import configure_matmul_precision
    from .lightning import seed_everything
    from .wandb import login as wandb_login

    configure_matmul_precision()
    seed_everything(config['seed'])
    wandb_login()
