cd ../../

nohup python run_experiment.py mnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1 >> nohup_logs/mnist/FedAvg.log 2>&1 &

nohup python run_experiment.py mnist local --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1 >> nohup_logs/mnist/local.log 2>&1 &

nohup python run_experiment.py mnist pFedMe --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --mu 1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer prox_sgd --seed 1233 --verbose 1 >> nohup_logs/mnist/pFedMe.log 2>&1 &

nohup python run_experiment.py mnist FedEM --n_learners 4 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1 >> nohup_logs/mnist/FedEM.log 2>&1 &

nohup python run_experiment.py mnist FedLG --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1 --include_keyword classifier >> nohup_logs/mnist/FedLG.log 2>&1 &

nohup python run_experiment.py mnist Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1 --exclude_keyword canonical --n_canonical 4 --client_lr 0.3 --client_scheduler multi_step >> nohup_logs/mnist/PPFL1.log 2>&1 &

nohup python run_experiment.py mnist Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1 --exclude_keyword canonical --n_canonical 4 --client_lr 0.5 --client_scheduler multi_step --interpolate >> nohup_logs/mnist/PPFL2.log 2>&1 &
