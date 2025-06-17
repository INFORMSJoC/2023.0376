#!/bin/bash  
cd ../../

# Fedavg
python -u train_pfl.py --pfl_algo fedavg --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256  --num_clients_per_round 10  --num_local_epochs 1 --client_lr 0.5  --client_scheduler const --server_optimizer sgd --server_lr 1.0 --server_momentum 0.0  --global_scheduler const_and_cut --global_lr_decay_factor 0.5 --global_lr_decay_every 500 --num_communication_rounds 2000 --logfilename ./checkpoint/pfl/pretrain_output/emnist_pretrain_2000 --savedir ./checkpoint/pfl/pretrain_model/emnist_pretrain_2000 

# PPFL1
python -u train_pfl.py --pfl_algo fedpop --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256 --num_clients_per_round 10 --num_local_epochs 1 --client_lr 0.5  --client_scheduler const --server_optimizer sgd --server_lr 1.0 --server_momentum 0.0 --global_scheduler const_and_cut --global_lr_decay_factor 0.5 --global_lr_decay_every 500 --num_communication_rounds 2000 --pretrain --n_canonical 10 --laplace_coef 1e-5 --logfilename ./checkpoint/pfl/pretrain_output/emnist_PPFL1_2000 --savedir ./checkpoint/pfl/pretrain_model/emnist_PPFL1_2000

# PPFL2

python -u train_pfl.py --pfl_algo fedpop --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256 --num_clients_per_round 10 --num_local_epochs 1 --client_lr 0.5  --client_scheduler const --server_optimizer sgd --server_lr 1.0 --server_momentum 0.0 --global_scheduler const_and_cut --global_lr_decay_factor 0.5 --global_lr_decay_every 500 --num_communication_rounds 2000 --pretrain --interpolate --n_canonical 10 --laplace_coef 1e-4 --logfilename ./checkpoint/pfl/pretrain_output/emnist_PPFL2_2000 --savedir ./checkpoint/pfl/pretrain_model/emnist_PPFL2_2000

