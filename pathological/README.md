This directory contains the implementation of experiments conducted under pathological settings , including:
- Data generation
- Training scripts
- Plotting scripts

# Data Preparation

Before running the experiments, please follow the instructions in the corresponding dataset's README.md file located under:

```
data/
├── cifar10/
│   └── README.md
├── covariate/
│   └── README.md
├── mnist/
│   └── README.md
└── synthetic/
    └── README.md
```

Each README.md provides detailed instructions on how to generate the federated dataset with controlled heterogeneity.

# Running Experiments

All experiment scripts are located in the `scripts/ directory`.

- To reproduce the results shown in Table 1 , please navigate to the corresponding dataset folder under scripts/ and execute the provided shell script.

For example, to run the MNIST experiment:
``` bash
cd scripts/mnist
./run_mnist.sh
```
- For experiments with varying levels of heterogeneity (e.g., reported in Table 2 ), please navigate to the corresponding dataset folder under scripts/ and execute the provided shell script.

For example, to run the MNIST experiment with $alpha=0.2$ :
``` bash
cd scripts/mnist
./mnist_alpha2e-1.sh
```

> These scripts use the nohup command to run experiments in the background, and the terminal prompt returns immediately — this does not mean the script has finished.
> Log files can be founded in the nohup_logs/ directory. 


# Plotting Results

After running the experiments, results will be saved in a structured format under the logs/ directory.

Visualization scripts used to reproduce the figures in the paper are stored in the plot/ directory. Each script corresponds directly to a specific figure and is named accordingly. 


# Acknowledge

This implementation builds upon the repository [FedEM](https://github.com/omarfoq/FedEM).