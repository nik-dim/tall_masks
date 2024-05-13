import os
from typing import Dict, Optional, Tuple

import torch
from omegaconf import DictConfig

from src.models.task_vectors import ImageEncoder, NonLinearTaskVector
from src.utils.tallmask_utils import construct_consensus_mask, construct_tall_mask, load_tall_mask
from src.utils.ties_utils import ties_merging
from src.utils.utils import (
    check_parameterNamesMatch,
    check_state_dicts_equal,
    state_dict_to_vector,
    topk_values_mask,
    vector_to_state_dict,
)
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path


def get_all_checkpoints(config: DictConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Retrieves all the checkpoints for the given configuration.

    Args:
        config (DictConfig): The configuration object containing the model location, datasets, and model name.

    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple containing two dictionaries.
            The first dictionary contains the checkpoints for each dataset in the configuration's validation datasets.
            The second dictionary contains the checkpoint for the zeroshot model.
    """

    model_dir = config.model_location
    print("I am getting out all the checkpoints")
    print("datasets:", config.DATASETS_VAL)
    print("model:", config.model)
    for dataset in config.DATASETS_VAL:
        path = get_finetuned_path(model_dir, dataset, model=config.model)
        if os.path.exists(path):
            print(f"{path} exists")
        else:
            print(f"{path} does not exist")

    params = {
        dataset: torch.load(get_finetuned_path(model_dir, dataset, model=config.model), map_location="cpu")
        for dataset in config.DATASETS_VAL
    }

    # convert dict to vector
    params = list(params.values())

    try:
        ptm_check = torch.load(get_zeroshot_path(model_dir, "MNISTVal", model=config.model), map_location="cpu")
    except:
        ptm_check = ImageEncoder(config.model).state_dict()
        torch.save(ptm_check, get_zeroshot_path(model_dir, "MNISTVal", model=config.model))

    return params, ptm_check


def create_task_vector(config: DictConfig) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Creates a task vector based on the given configuration.

    Args:
        config (DictConfig): The configuration for creating the task vector.

    Returns:
        Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]: A tuple containing the task vector and evaluation masks
            (if applicable).
    """

    ft_checks, ptm_check = get_all_checkpoints(config)
    check_parameterNamesMatch(ft_checks + [ptm_check])

    remove_keys = []

    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    # compute the task vector as {\theta_t - \theta_0}.
    tv_flat_checks = flat_ft - flat_ptm

    # check if the vectorized state dicts can be converted back to the original state dicts
    # covnert back the flat task vectors to state dict and see if the original and converted sd's are equal
    assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
    assert all(
        [
            check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i])
            for i in range(len(ft_checks))
        ]
    )
    print(f"MODEL: {config.model}, METHOD {config.method.name}")

    if config.method.name == "ties":
        # TIES Merging
        merge_func = "dis-mean"
        merged_tv = ties_merging(tv_flat_checks, reset_thresh=config.method.k, merge_func=merge_func)
    elif config.method.name in ["sum", "zeroshot", "average"]:
        # "sum" corresponds to Task Arithmetic (TA)
        # TA, zeroshot, weight average all construct the task vector with sum, but use different scaling factors.
        tv_flat_checks, _ = topk_values_mask(tv_flat_checks, K=config.method.k, return_mask=False)
        merged_tv = tv_flat_checks.sum(dim=0)
    elif config.method.name == "tall_mask":
        # construct multi-task vector
        if config.method.use_ties:
            print(f"Using TIES for constructing multi-task vector")
            merged_tv = ties_merging(tv_flat_checks, reset_thresh=20, merge_func=f"dis-sum")
        else:
            print(f"Using Task Arithmetic for constructing multi-task vector")
            tv_flat_checks, _ = topk_values_mask(tv_flat_checks, K=config.method.k, return_mask=False)
            merged_tv = tv_flat_checks.sum(dim=0)
        # get TALL masks
        if config.method.load_mask:
            # load tall masks directly from storage
            eval_masks = load_tall_mask(remove_keys, ptm_check, config)
        else:
            print(f"=== Constructing TALL Mask ===")
            # construct tall masks
            eval_masks = construct_tall_mask(
                tv_flat_checks, flat_ft, flat_ptm, merged_tv, ptm_check, remove_keys, config
            )
    elif config.method.name == "consensus":  # consensus merging
        # construct consensus mask (assuming the TALL masks have already been constructed)
        consensus_mask = construct_consensus_mask(ptm_check, config.method.prun_thre_k, config, remove_keys)
        # construct multi-task vector
        if config.method.use_ties:
            merged_tv = ties_merging(tv_flat_checks, reset_thresh=20, merge_func="dis-sum")
        else:
            tv_flat_checks, _ = topk_values_mask(
                tv_flat_checks, K=config.method.k, return_mask=False
            )  # top-k mag filtering
            merged_tv = tv_flat_checks.sum(dim=0)
        # apply the consensus mask to filter multi-task vector
        merged_tv = merged_tv * consensus_mask
    elif config.method.name == "mag_masking":
        # Magnitude masking baseline
        print(f"=== Using Magnitude Masking ===")
        merged_tv = tv_flat_checks.sum(dim=0)
        _, _, eval_masks = topk_values_mask(tv_flat_checks, K=config.method.k, return_mask=True)
        eval_masks = [vector_to_state_dict(mask, ptm_check, remove_keys=remove_keys) for mask in eval_masks]
        eval_masks = {key: value for key, value in zip(config.DATASETS, eval_masks)}
    else:
        raise ValueError(f"Method {config.method.name} not defined.")

    merged_tv_state_dict = vector_to_state_dict(merged_tv, ptm_check, remove_keys=remove_keys)

    task_vector = NonLinearTaskVector(model_name=config.model, vector=merged_tv_state_dict)
    print("Norm of task vector: ", task_vector.norm())

    if config.method.name not in ["tall_mask", "mag_masking"]:
        eval_masks = None

    return task_vector, eval_masks
