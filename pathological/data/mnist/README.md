# Introduction

This dataset construction script partitions the MNIST dataset into `n_tasks` clients with controlled domain heterogeneity.

- To generate the dataset as used in Table 1:

```python
python generate_data.py \
    --n_tasks 100 \
    --n_components 4 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --seed 12345
```

- To simulate different levels of heterogeneity using the Dirichlet distribution parameter alpha (e.g., the experiment in Table 2), use:
  
```python
python generate_alpha.py \
    --n_tasks 100 \
    --n_components 4 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --seed 12345 \
    --alpha 0.2
```

> Note: If the all_data directory already exists, please first remove the existing files using the following command:
> ```bash
> rm all_data/* -rf
> ```
> For alpha = 1.0, please extract or copy the pre-generated .tar files into the all_data folder.

```python
# for alpha = 0.2
python generate_alpha.py --n_tasks 100 --n_components 4 --s_frac 1.0 --tr_frac 0.8 --seed 12345 --alpha 0.2
```

```python
# for alpha = 0.5
python generate_alpha.py --n_tasks 100 --n_components 4 --s_frac 1.0 --tr_frac 0.8 --seed 12345 --alpha 0.5
```


# Argument Reference

- ```--n_tasks```: number of tasks/clients to split the data into
- ```--n_components```: number of domain-heterogeneous components 
- ```--s_frac```: fraction of the dataset to be used
- ```--tr_frac```: train set proportion for each task
- ```--test_tasks_frac```: fraction of test tasks
- ```--val_frac```: fraction of validation set (from train set)
- ```--seed```: seed to be used before random sampling of data