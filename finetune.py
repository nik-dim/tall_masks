import os
import time
from omegaconf import open_dict

import torch
import wandb

from src.datasets import get_dataloader, get_dataset, maybe_dictionarize
from src.eval.eval import eval_single_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.utils import initialize_wandb, parse_arguments
from src.utils.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.utils.utils import LabelSmoothing, cosine_lr
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path


def finetune(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    if is_main_process():
        initialize_wandb(args)

    train_dataset = args.train_dataset

    ft_path = get_finetuned_path(args.model_location, train_dataset, args.model)
    zs_path = get_zeroshot_path(args.model_location, train_dataset, args.model)

    if os.path.exists(zs_path) and os.path.exists(ft_path):
        if is_main_process():
            print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    image_encoder = ImageEncoder(args.model)

    classification_head = get_classification_head(args, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The toal number of trainable parameters is {num_params/1e6:.2f}M")

    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(train_dataset, preprocess_fn, location=args.data_location, batch_size=args.batch_size)
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)

    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True, output_device=rank
    )

    print("hello from process", rank)

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(
        optimizer, args.lr, args.warmup_length, args.epochs * num_batches // args.num_grad_accumulation
    )

    # Saving zero-shot model
    if is_main_process():
        ckpdir = os.path.join(args.save_dir, train_dataset)
        os.makedirs(ckpdir, exist_ok=True)
        model_path = get_zeroshot_path(args.model_location, train_dataset, args.model)
        ddp_model.module.image_encoder.save(model_path)

    for epoch in range(args.epochs):
        ddp_model.train()

        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = i // args.num_grad_accumulation + epoch * num_batches // args.num_grad_accumulation

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time

            logits = ddp_model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0 and is_main_process():
                print("Saving checkpoint.")
                model_path = get_finetuned_path(args.model_location, train_dataset, args.model).replace(
                    ".pt", f"_{step}.pt"
                )
                ddp_model.module.image_encoder.save(model_path)

            if step % print_every == 0 and ((i + 1) % args.num_grad_accumulation == 0) and is_main_process():
                percent_complete = 100 * i / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t",  # noqa: E501
                    flush=True,
                )
                wandb.log(
                    {
                        f"{train_dataset}/train/loss": loss.item(),
                        "train/data_time": data_time,
                        "train/batch_time": batch_time,
                    }
                )

    if is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = ddp_model.module.image_encoder
        test_accuracy = eval_single_dataset(image_encoder, train_dataset, args)

    if is_main_process():
        ft_path = get_finetuned_path(args.model_location, train_dataset, args.model)
        zs_path = get_zeroshot_path(args.model_location, train_dataset, args.model)

        image_encoder.save(ft_path)
        return zs_path, ft_path

    cleanup_ddp()


if __name__ == "__main__":

    # uncomment all the datasets for fine-tuning
    train_datasets = [
        "MNIST",
        # "Cars",
        # "DTD",
        # "EuroSAT",
        # "GTSRB",
        # "RESISC45",
        # "SUN397",
        # "SVHN",
        # "CIFAR100",
        # "STL10",
        # "Flowers102",
        # "OxfordIIITPet",
        # "FER2013",
        # "PCAM",
        # "FashionMNIST",
        # "CIFAR10",
        # "Food101",
        # "RenderedSST2",
        # "KMNIST",
        # "EMNIST",
    ]
    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
        "CIFAR10": 6,
        "CIFAR100": 6,
        "STL10": 60,
        "Food101": 4,
        "Flowers102": 147,
        "FER2013": 10,
        "PCAM": 1,
        "OxfordIIITPet": 82,
        "RenderedSST2": 39,
        "EMNIST": 2,
        "FashionMNIST": 5,
        "KMNIST": 5,
    }

    for dataset in train_datasets:
        args = parse_arguments()
        args.lr = 1e-5
        args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"
        args.save_dir = os.path.join(args.model_location, args.model)

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        args.batch_size = 64 if args.model == "ViT-L-14" else 128
        args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
        # finetune(0, args)
