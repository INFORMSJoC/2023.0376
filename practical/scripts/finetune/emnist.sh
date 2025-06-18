#!/bin/bash  
cd ../../

for num_epoch in 1 2 3 4 5
do 
python -u train_finetune.py --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256 --lr 1e-2 --pretrained_model_path ./checkpoint/pfl/pretrain_model/emnist_pretrain_2000/checkpoint.pt --seed 1 --personalize_on_client finetune --num_epochs_personalization "${num_epoch}" --logfilename "./checkpoint/pfl/finetune_output/emnist_finetune_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/emnist_finetune_${num_epoch}" 

python -u train_finetune.py --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256 --lr 1e-2 --pretrained_model_path ./checkpoint/pfl/pretrain_model/emnist_pretrain_2000/checkpoint.pt --seed 1 --personalize_on_client finetune --client_var_l2_reg_coef 0.1 --client_var_prox_to_init --num_epochs_personalization "${num_epoch}" --logfilename "./checkpoint/pfl/finetune_output/emnist_ditto_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/emnist_ditto_${num_epoch}"

python -u train_finetune.py --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256 --lr 1e-2 --pretrained_model_path ./checkpoint/pfl/train_model/emnist_pfedme/checkpoint.pt --seed 1 --personalize_on_client finetune --client_var_l2_reg_coef 0.1 --client_var_prox_to_init --num_epochs_personalization "${num_epoch}" --logfilename "./checkpoint/pfl/finetune_output/emnist_pfedme_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/emnist_pfedme_${num_epoch}"

python -u train_finetune.py --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256 --lr 1e-2  --pretrained_model_path ./checkpoint/pfl/train_model/emnist_pfl/checkpoint.pt --seed 1 --personalize_on_client adapter --num_epochs_personalization "${num_epoch}" --logfilename "./checkpoint/pfl/finetune_output/emnist_pfl_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/emnist_pfl_${num_epoch}" 

python -u train_finetune.py --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256  --lr 1e-2  --pretrained_model_path ./checkpoint/pfl/pretrain_model/emnist_PPFL1_2000/checkpoint.pt --seed 1 --personalize_on_client canonical  --num_epochs_personalization "${num_epoch}" --logfilename "./checkpoint/pfl/finetune_output/emnist_PPFL1_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/emnist_PPFL1_${num_epoch}" 

python -u train_finetune.py --dataset emnist  --model_name resnet_gn --train_batch_size 32 --eval_batch_size 256  --lr 1e-2  --pretrained_model_path ./checkpoint/pfl/pretrain_model/emnist_PPFL2_2000/checkpoint.pt --seed 1 --personalize_on_client canonical  --interpolate --num_epochs_personalization "${num_epoch}" --logfilename "./checkpoint/pfl/finetune_output/emnist_PPFL2_${num_epoch}" --savedir "./checkpoint/pfl/finetune_model/emnist_PPFL2_${num_epoch}" 
done