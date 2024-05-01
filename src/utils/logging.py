import os
from argparse import Namespace
from pprint import pprint

import wandb

from src.utils.distributed import is_main_process


def initialize_wandb(args, disabled=True):
    if disabled:
        # for debugging
        wandb.init(config=args, mode="disabled")
    else:
        wandb.init(config=args)

    if wandb.run is not None:
        INVALID_PATHS = ["__old__", "checkpoints", "logs", "outputs", "results", "wandb"]
        wandb.run.log_code(
            exclude_fn=lambda path: any(
                [path.startswith(os.path.expanduser(os.getcwd() + "/" + i)) for i in INVALID_PATHS]
            )
        )
    return wandb


def wandb_log(dictionary: dict):
    if is_main_process():
        wandb.log(dictionary)
