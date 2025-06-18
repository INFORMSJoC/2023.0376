#!/bin/bash
cd ../../


nohup python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 >> nohup_logs/cifar10/Fedavg.log 2>&1 &

nohup python run_experiment.py cifar10 local --n_learners 1 --n_rounds 201 --bz 128 --lr 0.03 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 >> nohup_logs/cifar10/local.log 2>&1 & 

nohup python run_experiment.py cifar10 Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --exclude_keyword canonical --n_canonical 3 --client_lr 0.5 --client_scheduler multi_step >> nohup_logs/cifar10/PPFL1.log 2>&1 &

nohup python run_experiment.py cifar10 Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --exclude_keyword canonical --n_canonical 3 --client_lr 0.5 --client_scheduler multi_step --interpolate >> nohup_logs/cifar10/PPFL2.log 2>&1 &

nohup python run_experiment.py cifar10 pFedMe --n_learners 1 --n_rounds 201 --bz 128 --lr 0.01 --mu 30.0 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 >> nohup_logs/cifar10/pFedMe.log 2>&1 &

nohup python run_experiment.py cifar10 FedEM --n_learners 3 --n_rounds 201 --bz 128 --lr 0.03 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 >> nohup_logs/cifar10/FedEM.log 2>&1 &

nohup python run_experiment.py cifar10 FedLG --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1235 --verbose 1 --include_keyword classifier >> nohup_logs/cifar10/FedLG.log 2>&1 &
