import os
import json
from pprint import pprint

import numpy as np

from src.eval.eval import eval_single_dataset
from src.models.task_vectors import NonLinearTaskVector
from src.utils.args import parse_arguments
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path

args = parse_arguments()
args.save_dir = os.path.join(args.model_location, args.model)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print("*" * 100)

pprint(args.__dict__, width=1)
accuracies = {}

# load pretrained checkpoint
pretrained_checkpoint = get_zeroshot_path(args.model_location, "MNIST", args.model)

# evaluate each task sequentially
for dataset in [
    "MNIST",
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "RESISC45",
    "SUN397",
    "SVHN",
    "PCAM",
    "CIFAR100",
    "STL10",
    "OxfordIIITPet",
    "Flowers102",
    "FER2013",
    "CIFAR10",
    "Food101",
    "RenderedSST2",
    "EMNIST",
    "FashionMNIST",
    "KMNIST",
]:

    print("\n" * 3)
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    # load finetuned checkpoint
    finetuned_checkpoint = get_finetuned_path(args.model_location, dataset, args.model)
    task_vector = NonLinearTaskVector(args.model, pretrained_checkpoint, finetuned_checkpoint)
    
    if args.finetuning_mode == "none":
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
        save_file = f"single_task_zeroshot_accuracies.json"
    elif args.finetuning_mode == "standard":
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        save_file = f"single_task_nonlinear_ft_accuracies.json"
    
    for split in ["val", "test"]:
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        if eval_dataset not in accuracies:
            accuracies[eval_dataset] = {}
        accuracies[eval_dataset] = eval_single_dataset(image_encoder, eval_dataset, args)["top1"]
        print()

directory = f"results/single_task/{args.model}"
if not os.path.exists(directory):
    os.makedirs(directory)
save_path = os.path.join(directory, save_file)
with open(save_path, "a+") as f:
    f.write(json.dumps(accuracies, sort_keys=False, indent=4) + '\n')

pprint(accuracies, width=1)
print("File saved at: ", save_path)