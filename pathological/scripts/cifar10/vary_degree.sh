cd ../../

for num in 2 4 5
do
nohup python run_experiment.py cifar10 Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1 --exclude_keyword canonical --n_canonical ${num} --client_lr 0.5 --client_scheduler multi_step >> nohup_logs/cifar10/PPFL1_${num}.log 2>&1 &
done
