#!/bin/bash  
cd ../../

for num_epoch in 1 2 3 4 5
do 

python -u train_finetune.py --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 --clip_grad_norm  --arch_size mini --lr 0.1 --pretrained_model_path ./checkpoint/pfl/pretrain_model/stackoverflow_pretrain_1000/checkpoint.pt --num_epochs_personalization "${num_epoch}" --seed 1 --personalize_on_client finetune --logfilename "./checkpoint/pfl/finetune_output/stackoverflow_finetune_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/stackoverflow_finetune_${num_epoch}"

python -u train_finetune.py --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 --clip_grad_norm  --arch_size mini --lr 0.1 --pretrained_model_path ./checkpoint/pfl/pretrain_model/stackoverflow_pretrain_1000/checkpoint.pt --num_epochs_personalization "${num_epoch}" --client_var_l2_reg_coef 0.001 --client_var_prox_to_init --seed 1 --personalize_on_client finetune --logfilename "./checkpoint/pfl/finetune_output/stackoverflow_ditto_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/stackoverflow_ditto_${num_epoch}"

python -u train_finetune.py --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 --clip_grad_norm  --arch_size mini --lr 0.1 --pretrained_model_path ./checkpoint/pfl/train_model/stackoverflow_pfedme/checkpoint.pt --num_epochs_personalization "${num_epoch}" --client_var_l2_reg_coef 1e-4 --client_var_prox_to_init --seed 1 --personalize_on_client finetune --logfilename "./checkpoint/pfl/finetune_output/stackoverflow_pfedme_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/stackoverflow_pfedme_${num_epoch}"

python -u train_finetune.py --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 --clip_grad_norm  --arch_size mini --lr 0.1 --pretrained_model_path ./checkpoint/pfl/train_model/stackoverflow_pfl/checkpoint.pt --num_epochs_personalization "${num_epoch}" --seed 1 --personalize_on_client tr_layer --layers_to_finetune 3 --logfilename "./checkpoint/pfl/finetune_output/stackoverflow_pfl_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/stackoverflow_pfl_${num_epoch}"

python -u train_finetune.py --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 --clip_grad_norm --arch_size mini --lr 0.1 --pretrained_model_path ./checkpoint/pfl/pretrain_model/stackoverflow_PPFL1_1000/checkpoint/pt --seed 1 --num_epochs_personalization "${num_epoch}" --personalize_on_client canonical --optimizer md  --logfilename "./checkpoint/pfl/finetune_output/stackoverflow_PPFL1_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/stackoverflow_PPFL1_${num_epoch}"

python -u train_finetune.py --dataset stackoverflow --train_batch_size 64 --eval_batch_size 1024 --clip_grad_norm --arch_size mini --lr 0.1 --pretrained_model_path ./checkpoint/pfl/pretrain_model/stackoverflow_PPFL2_1000/checkpoint/pt --interpolate --seed 1 --num_epochs_personalization "${num_epoch}" --personalize_on_client canonical --optimizer md  --logfilename "./checkpoint/pfl/finetune_output/stackoverflow_PPFL2_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/stackoverflow_PPFL2_${num_epoch}"
done