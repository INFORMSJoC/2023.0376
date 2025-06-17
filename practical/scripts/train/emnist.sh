#!/bin/bash  
cd ../../

python -u train_pfl.py --pfl_algo fedalt --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256 --num_communication_rounds 500 --num_clients_per_round 10 --num_local_epochs 1 --client_scheduler const --server_optimizer sgd --server_lr 1.0 --server_momentum 0.0 --client_lr 0.01 --global_scheduler const_and_cut --global_lr_decay_factor 0.5 --global_lr_decay_every 500 --pretrained_model_path ./checkpoint/pfl/pretrain_model/emnist_pretrain_2000/checkpoint.pt --seed 1 --personalize_on_client adapter --logfilename ./checkpoint/pfl/train_output/emnist_pfl --savedir ./checkpoint/pfl/train_model/emnist_pfl

python -u train_pfl.py --pfl_algo pfedme --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256 --num_communication_rounds 500 --num_clients_per_round 10 --num_local_epochs 1 --client_scheduler const --server_optimizer sgd --server_lr 1.0 --server_momentum 0.0 --client_lr 0.01 --global_scheduler const_and_cut --global_lr_decay_factor 0.5 --global_lr_decay_every 500 --pretrained_model_path /checkpoint/pfl/pretrain_model/emnist_pretrain_2000/checkpoint.pt --seed 1 --pfedme_l2_reg_coef 0.1 --logfilename ./checkpoint/pfl/train_output/emnist_pfedme --savedir ./checkpoint/pfl/train_model/emnist_pfedme

