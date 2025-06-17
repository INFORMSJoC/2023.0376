#! /bin/bash

cd ../../

nohup python run_experiment.py covariate FedAvg --n_learners 1 --n_rounds 100 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 >> nohup_logs/covariate/Fedavg_covariate.log 2>&1 &

nohup python run_experiment.py covariate local --n_learners 1 --n_rounds 100 --bz 128 --lr 0.03 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 >> nohup_logs/covariate/local_covariate.log 2>&1 &

nohup python run_experiment.py covariate pFedMe --n_learners 1 --n_rounds 100 --bz 128 --lr 0.03 --mu 1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 >> nohup_logs/covariate/pFedMe_covariate.log 2>&1 &

nohup python run_experiment.py covariate FedEM --n_learners 5 --n_rounds 100 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1  >> nohup_logs/covariate/FedEM_covariate.log 2>&1 &

nohup python run_experiment.py covariate FedLG --n_learners 1 --n_rounds 100 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --include_keyword classifier >> nohup_logs/covariate/FedLG_covariate.log 2>&1 &

nohup python run_experiment.py covariate Fedpop --n_learners 1 --n_rounds 100 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --exclude_keyword canonical --n_canonical 4 --client_lr 0.3 --client_scheduler multi_step  >> nohup_logs/covariate/PPFL1_covariate.log 2>&1 &

nohup python run_experiment.py covariate Fedpop --n_learners 1 --n_rounds 100 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --exclude_keyword canonical --n_canonical 4 --client_lr 0.5 --client_scheduler multi_step --interpolate  >> nohup_logs/covariate/PPFL2_covariate.log 2>&1 &