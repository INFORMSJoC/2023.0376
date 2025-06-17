# Modified by Hao Di on 2025-06-17
# Based on original work licensed under the Apache License, Version 2.0

"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
from utils.utils import *
from utils.constants import *
from utils.args import *

from torch.utils.tensorboard import SummaryWriter

from omegaconf import OmegaConf


def init_clients(args_, root_path, logs_dir):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root /logs/data/method_seed/train
    :return: List[Client]
    """
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators, task_id_map =\
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            root_path=root_path,
            batch_size=args_.bz,
            is_validation=args_.validation
        )

    print("===> Initializing clients..")
    print(f'the length of train_iterators is {len(train_iterators)}, val_iterators is {len(val_iterators)}, test_iterators is {len(test_iterators)}')
    
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue
        learners_ensemble =\
            get_learners_ensemble(
                n_learners=args_.n_learners,
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                include_keyword=args_.include_keyword,
                exclude_keyword=args_.exclude_keyword,
                canonical=args_.canonical,
                n_canonical=args_.n_canonical,
                interpolate=args_.interpolate
            )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients,
            client_lr = args_.client_lr,
            client_scheduler = args_.client_scheduler,
        )

        clients_.append(client)

    return clients_, task_id_map


def run_experiment(args_):
    # print the config
    cfg = OmegaConf.create(vars(args_))
    print(f"the hyperparameter is \n{OmegaConf.to_yaml(cfg)}")


    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)  # 

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))  # logs_dir = /logs/dataset/method_seed

    print("==> Clients initialization..")
    clients, task_id_map = init_clients(args_, root_path=os.path.join(data_dir, "train"), logs_dir=os.path.join(logs_dir, "train"))

    print("==> Test Clients initialization..")
    test_clients, test_task_map = init_clients(args_, root_path=os.path.join(data_dir, "test"),
                                logs_dir=os.path.join(logs_dir, "test"))

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            include_keyword=args_.include_keyword,
            exclude_keyword=args_.exclude_keyword,
            canonical=args_.canonical,
            n_canonical=args_.n_canonical,
            interpolate=args_.interpolate
        )

    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    similarity_path = None
    if args_.method == "Fedpop":
        similarity_path = os.path.join(data_dir, 'similarity.npy')

    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed,
            laplace_coef=args_.laplace_coef,
            similarity_path=similarity_path,
            saved_dir=logs_dir,
            task_id_map = task_id_map,  # new
        )

    print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    aggregator.update_clients()
    while current_round <= args_.n_rounds:

        aggregator.mix()

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round

    if "save_dir" in args_:
        save_dir = os.path.join(args_.save_dir)

        os.makedirs(save_dir, exist_ok=True)
        aggregator.save_state(save_dir)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    if args.include_keyword and args.exclude_keyword:
        raise 'one of include_keyword and excluded_keyword should be none'

    if args.method == 'Fedpop':
        args.canonical = True
    else:
        args.canonical = False
    run_experiment(args)
