#!/bin/bash
cd ../../

nohup python run_experiment.py synthetic FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --input_dimension 150 --output_dimension 2 >> nohup_logs/synthetic/Fedavg.log 2>&1 &

nohup python run_experiment.py synthetic local --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --input_dimension 150 --output_dimension 2 >> nohup_logs/synthetic/local.log 2>&1 &

nohup python run_experiment.py synthetic pFedMe --n_learners 1 --n_rounds 201 --bz 128 --lr 0.1 --mu 1.0 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --input_dimension 150 --output_dimension 2 >> nohup_logs/synthetic/pFedMe.log 2>&1 &

nohup python run_experiment.py synthetic clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --input_dimension 150 --output_dimension 2 >> nohup_logs/synthetic/clustered.log 2>&1 &

nohup python run_experiment.py synthetic FedEM --n_learners 3 --n_rounds 201 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --input_dimension 150 --output_dimension 2 >> nohup_logs/synthetic/FedEM.log 2>&1 &

# LG is same as Fedavg

nohup python run_experiment.py synthetic Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --exclude_keyword canonical --n_canonical 4 --client_lr 0.9 --client_scheduler multi_step --input_dimension 150 --output_dimension 2 >> nohup_logs/synthetic/PPFL1.log 2>&1 &

nohup python run_experiment.py synthetic Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --exclude_keyword canonical --n_canonical 3 --client_lr 0.9 --client_scheduler multi_step --input_dimension 150 --output_dimension 2 --interpolate >> nohup_logs/synthetic/PPFL2.log 2>&1 &