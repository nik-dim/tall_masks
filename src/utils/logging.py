import os
import json
import hydra
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

def log_results(final_results, args):
    if args.method.name == "tall_mask":
        mask_suffix = f"tall_mask_ties" if args.method.use_ties else f"tall_mask_ta"
    elif args.method.name == "mag_masking":
        mask_suffix = "mag_mask"
    elif args.method.name == "consensus":
        mask_suffix = (
            f"k_{args.method.mask_agree_k}_ties" if args.method.use_ties else f"k_{args.method.mask_agree_k}_ta"
        )
    else:
        mask_suffix = ""

    save_file = f"results/merging/{args.model}_{args.num_tasks}tasks_{args.method.full_name}_nonlinear_additions_{mask_suffix}.json"

    with open(save_file, "w") as f:
        json.dump(final_results, f, indent=4)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    hydra_save_file = f"{args.method.full_name}_nonlinear_additions.json"
    hydra_save_file = os.path.join(hydra_dir, hydra_save_file)
    json.dump(final_results, open(hydra_save_file, "w"), indent=4)

    print("saved results to: ", save_file)
    print("saved results to: ", hydra_save_file)
    artifact = wandb.Artifact(name="final_results", type="results")
    artifact.add_file(save_file)
    wandb.log_artifact(artifact)