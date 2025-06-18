[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# PPFL: A Personalized Federated Learning Framework for Heterogeneous Population

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

##  Setup
This repository is built in PyTorch 1.10.0 and tested on Ubuntu 22.04 environment (Python3.8.10, CUDA11.3).
Follow the instructions below to set up the environment for this project.

1. Create and activate a Conda environment
```
conda create --name py3.8 python=3.8.10 
conda activate py3.8
```
1. Install required packages
```
pip install -r requirements.txt
```
> Note on jaxlib==0.1.55
> The version of tensorflow-federated used in this project depends on jaxlib==0.1.55, which is no longer available on PyPI. You can find the corresponding wheel file for your platform at: https://storage.googleapis.com/jax-releases/jax_releases.html

## Code Structure

There are two main directories in this project:

- pathological/ : Contains the source code for experiments conducted under pathological settings.
- practical/ : Contains the source code for experiments designed for real-world or practical scenarios.
  
For detailed information about each folder, including data structure, configuration, and usage instructions, please refer to the README.md file within each directory.

```
├── pathological/                
│   ├── dataset/
│       ├── cifar10/  
│       └── covariate/           
│       └── mnist/
│       └── synthetic/
│   ├── learners/                 
│   └── plot/                  
│   └── scripts/
│       ├── cifar10/  
│       └── covariate/           
│       └── mnist/
│       └── synthetic/
└── practical/
│   ├── dataset_statistics/
│   ├── pfl/
│       ├── data/  
│       ├── models/  
│       ├── optim/  
│   ├── scripts/
│       ├── finetune/  
│       └── pretrain/           
│       └── train/
```

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0376

https://doi.org/10.1287/ijoc.2023.0376.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{ppfl2023,
  author    = {Di, Hao and Yang, Yi and Ye, Haishan and Xiang, Yuchang},
  title     = {PPFL: A Personalized Federated Learning Framework for Heterogeneous Population},
  year      = {2025},
  publisher = {INFORMS Journal on Computing},
  doi       = {10.1287/ijoc.2023.0376},
  url       = {https://github.com/INFORMSJoC/2023.0376},
  note      = {Available for download at https://github.com/INFORMSJoC/2023.0376},
}  
```

## License Clarification

This repository is released under the MIT License. The code in the `pathological/` subdirectory is based on [FedEM](https://github.com/omarfoq/FedEM) released under the Apache License 2.0, and that license is retained within the folder.