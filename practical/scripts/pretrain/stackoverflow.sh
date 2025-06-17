#!/bin/bash  
cd ../../

# Fedavg
python -u train_pfl.py --pfl_algo fedavg --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 --num_clients_per_round 50 --num_local_epochs 1 --clip_grad_norm  --log_test_every_n_rounds 100 --max_num_clients_for_logging 1000 --arch_size mini  --server_optimizer adam --server_lr 5e-4 --client_scheduler const --client_lr 1 --client_optimizer sgd --global_scheduler linear --global_warmup_fraction 0.1 --num_communication_rounds 1000 --logfilename  ./checkpoint/pfl/pretrain_output/stackoverflow_pretrain_1000  --savedir ./checkpoint/pfl/pretrain_model/stackoverflow_pretrain_1000

# PPFL1
python -u train_pfl.py --pfl_algo fedpop --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 --num_clients_per_round 50 --num_local_epochs 1 --clip_grad_norm  --log_test_every_n_rounds 100 --max_num_clients_for_logging 1000 --arch_size mini --server_optimizer adam --server_lr 5e-4 --client_scheduler const --client_lr 1 --client_optimizer sgd --global_scheduler linear --global_warmup_fraction 0.1 --pretrain  --num_communication_rounds 1000  --n_canonical 5 --laplace_coef 1e-5 --logfilename ./checkpoint/pfl/pretrain_output/stackoverflow_PPFL1_1000 --savedir ./checkpoint/pfl/pretrain_model/stackoverflow_PPFL1_1000

# PPFL2
python -u train_pfl.py --pfl_algo fedpop --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 --num_clients_per_round 50 --num_local_epochs 1 --clip_grad_norm  --log_test_every_n_rounds 100 --max_num_clients_for_logging 1000 --arch_size mini --server_optimizer adam --server_lr 5e-4 --client_scheduler const --client_lr 1 --client_optimizer sgd --global_scheduler linear --global_warmup_fraction 0.1 --pretrain  --interpolate --num_communication_rounds 1000  --n_canonical 2 --laplace_coef 1e-5 --logfilename ./checkpoint/pfl/pretrain_output/stackoverflow_PPFL2_1000 --savedir ./checkpoint/pfl/pretrain_model/stackoverflow_PPFL2_1000