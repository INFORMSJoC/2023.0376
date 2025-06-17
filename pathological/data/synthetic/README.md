# Introduction

This dataset construction script generates a synthetic federated learning dataset , where each client's data is sampled from a mixture of n_components underlying distributions. 

To generate the synthetic dataset with default parameters (as used in our experiments):

```python
python generate_data.py \
    --n_tasks 300 \
    --n_classes 2 \
    --n_components 3 \
    --dimension 150 \
    --noise_level 0.1 \
    --n_test 5000 \
    --alpha 0.4 \
    --seed 12345
```

> Note: If the all_data directory already exists, please first remove its contents before running the command: 
> ```bash
> rm -rf all_data/*
> ```

# Argument Reference

- ```--n_tasks```: number of tasks/clients to split the data into;
- ```--n_classes``` : number of classes in the generated dataset;
- ```--n_components```: number of mixture components;
- ```--dimension```: dimensionality of the synthetic data;
- ```--noise_level``` : proportion of noise added to feature;
- ```--n_test```: size of the test set;
- ```--alpha```: dirichlet concentration parameter; 
- ```--uniform_marginal```: flag indicating whether all tasks should share the same marginal distribution;
- ```--train_tasks_frac```: fraction of tasks used for training (the rest are used for testing);
- ```--seed``` : random seed for reproducibility