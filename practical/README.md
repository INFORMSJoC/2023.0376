This directory contains the implementation of experiments conducted under practical settings.
The code supports data generation and training on two real-world federated datasets: EMNIST and StackOverflow.

# Data Preparation

To generate the datasets, please follow the instructions in `dataset_statistics/README.md`.

# Running Experiments

All experiment scripts are located in the scripts/ directory.
The pipeline follows the structure used in [PFL: Partial Personalization in Federated Learning](https://proceedings.mlr.press/v162/pillutla22a.html):

1. Pretraining

Pretrain a global model using `FedAvg` or `PPFL` by specifying the appropriate `logfilename` and `save_dir`:
```bash
cd scripts/pretrain
./run_pretrain.sh
```


2. Training
For personalization methods such as `FedAlt` and `pFedMe`, run the corresponding training script using the pretrained model:

```bash
cd scripts/train
./run_pfedme.sh
```

3. Local Fine-tune

Run the relevant fine-tuning script to obtain the final results.  
> Note: Except for `PPFL` and `FedAvg`, all other methods fine-tune based on the model obtained from the training stage.
```bash
cd scripts/finetune
./run_finetune.sh
```

# Acknowledge

This implementation builds upon the repository [pfl](https://github.com/facebookresearch/FL_partial_personalization).