# README

This is the source code to reproduce the experiments for "[Localizing Task Information for Improved Model Merging and Compression](https://arxiv.org/abs/tbd)" by Ke Wang*, Nikolaos Dimitriadis*, Guillermo Ortiz-Jimenez, Francois Fleuret, and Pascal Frossard.

Our paper identifies that the task-specific knowledge is preserved after mering, and proposed a method named TALL mask to localize them.
Based on TALL mask, we proposed:
1) a compression scheme which utilizes TALL mask to recover single-task fine-tuned performance for each task
2) a merging algorithm which removes catastrophic and selfish weights to improve model merging performance

![](figures/illustration.png)

## Dependencies

To run the code, please install all its dependencies:
```sh
conda env create
conda activate tall-masks
```

## Checkpoints
We provide the checkpoints, as well as the generated task-specific masks we used in the paper in [this link](https://drive.google.com/drive/folders/15ParSng4d5xSdaWdBFsg1617zPXT8Dae?usp=sharing).

## Finetuning
The script `finetune.py` can be used to reproduce the training protocol we used to fine-tune our models on all our downstream tasks.
```sh 
# Finetune on 2 GPUs
python finetune.py --model=ViT-B-32 --world-size=2 
```

## Evaluation

### Single-task evaluation
You can evaluate the performance of the fine-tuned weights on each single task by running
```sh 
# Evaluate pre-trained models.
python eval_single_task.py --model=ViT-B-32 --finetuning-mode=none

# Evaluate non-linearly fine-tuned models.
python eval_single_task.py --model=ViT-B-32 --finetuning-mode=standard
```

### Model merging evaluation

Evaluation is performed with Hydra. The main script is `main.py` and the configuration files are in the `config/` folder. 


##### Evaluate with baseline model merging methods:
```bash
# Evaluate with Task Arithmetic
python main.py model=ViT-B-32 method="sum" 

# Evaluate with Ties-merging
python main.py model=ViT-B-32 method="ties" method.k=20
```
##### Evaluate with TALL mask + model merging methods:
```bash
# Evaluate with Tall mask + Task Arithmetic (load tall masks from storage)
python main.py model=ViT-B-32 method="tall_mask" method.load_mask=True

# Evaluate with Tall mask + Task Arithmetic (construct tall masks from scratch)
python main.py model=ViT-B-32 method="tall_mask"

# Evaluate with Tall mask + Ties-merging (load tall masks from storage)
python main.py model=ViT-B-32 method="tall_mask" method.use_ties=True method.ties_agg="sum" method.load_mask=True

# Evaluate with Tall mask + Ties-merging (construct tall masks from scratch)
python main.py model=ViT-B-32 method="tall_mask" method.use_ties=True method.ties_agg="sum"
```
##### Evaluate with Consensus Merging (after constructing TALL masks):
``` bash
# Evaluate with Consensus Task Arithmetic
python main.py model=ViT-B-32 method="consensus" method.mask_agree_k=2

# Evaluate with Consensus Ties-merging
python main.py model=ViT-B-32 method="consensus" method.mask_agree_k=2 method.use_ties=True
```

Note that you can set different number of tasks by setting `num_tasks`.

The results are saved in the `results/` folder. 

## Datasets
Most datasets being used should be downloaded automatically with torchvision or huggingface. For the datasets requiring manual preparation, please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1).

## Reference
If you find this code useful, please cite the following paper:
```bibtex

TBD

```

