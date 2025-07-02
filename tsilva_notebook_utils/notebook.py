from .lightning import seed_everything
from .torch import configure_matmul_precision
from .wandb import login as wandb_login


def setup_env(config):

    configure_matmul_precision()
    seed_everything(config['seed'])
    wandb_login()
