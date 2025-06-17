cd ../../

for num in 3 5 6
do
nohup python run_experiment.py mnist Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1 --exclude_keyword canonical --n_canonical ${num} --client_lr 0.3 --client_scheduler multi_step >> nohup_logs/mnist/PPFL1_${num}.log 2>&1 &
done