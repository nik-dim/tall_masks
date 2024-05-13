import json
import os

import torch
import wandb
from omegaconf import open_dict

from src.eval.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.utils.tallmask_utils import find_optimal_mask
from src.utils.utils import find_optimal_coef
from src.utils.logging import log_results
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path, get_single_task_accuracies_path


def perform_eval_with_merged_vector(args, task_vector, eval_masks=None):
    assert task_vector is not None, "Task vector should not be None."
    if eval_masks is not None:
        assert args.method.name in ["tall_mask", "mag_masking"]
    with open_dict(args):
        args.save_dir = os.path.join(args.model_location, args.model)

    ft_accuracies_path = get_single_task_accuracies_path(args.model)
    pretrained_checkpoint = get_zeroshot_path(args.model_location, "MNIST", args.model)

    with open_dict(args):
        with open(ft_accuracies_path) as f:
            args.finetuning_accuracies = json.load(f)
        args.eval_datasets = args.DATASETS_VAL
        args.control_dataset = None

    # evaluate on validation set
    val_metrics = evaluate_task_vector(task_vector, pretrained_checkpoint, args, eval_masks=eval_masks)

    if args.method.name == "tall_mask":
        if args.method.load_mask:
            best_masks_for_test = eval_masks
            best_val_metrics = val_metrics
        else:
            # find the best mask individually for each task based on validation accuracy
            best_masks_for_test, best_val_metrics = find_optimal_mask(val_metrics, eval_masks, args, save_masks=True)
    elif args.method.name == "mag_masking":
        best_masks_for_test = eval_masks
        best_val_metrics = val_metrics[1.0]
    else:
        # find scaling factor alpha based on validation accuracy (for Task Arithmetic, TIES, Consensus Merging)
        optimal_coef = find_optimal_coef(val_metrics, metric="avg_normalized_top1", minimize=False)
        best_val_metrics = val_metrics[optimal_coef]

    print("\n" * 2)

    # Evaluate on the test set with the optimal coefficients / masks
    with open_dict(args):
        args.eval_datasets = args.DATASETS

    if args.method.name in ["tall_mask", "mag_masking"]:
        test_metrics = evaluate_task_vector_at_coef(
            task_vector, pretrained_checkpoint, args, 1.0, eval_masks=best_masks_for_test
        )
    else:
        test_metrics = evaluate_task_vector_at_coef(
            task_vector, pretrained_checkpoint, args, float(optimal_coef), eval_masks=None
        )

    print("=" * 100)
    print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
    print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
    final_results = {"test": test_metrics, "val": val_metrics, "val_best": best_val_metrics}

    log_results(final_results, args)

    return final_results
