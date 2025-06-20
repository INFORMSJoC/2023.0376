# Modified by Hao Di on 2025-06-17
# Based on original work licensed under the Apache License, Version 2.0

from http import client
from models import *
from datasets import *
from learners.learner import *
from learners.learners_ensemble import *
from client import *
from aggregator import *

from .optim import *
from .metrics import *
from .constants import *
from .decentralized import *

from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from tqdm import tqdm

from collections import OrderedDict


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment

    :param experiment_name: name of the experiment
    :return: str

    """
    if experiment_name == 'covariate':
        data_dir = os.path.join("data", experiment_name)
    else:
        data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def get_learner(
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None,
        include_keyword=None,
        exclude_keyword=None,
        canonical=False,
        n_canonical=0,
        interpolate=False,
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, used for synthetic and har dataset
    :param output_dim: output_dimension; used for synthetic and har dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: Learner

    """
    torch.manual_seed(seed)
    used_mirror = False

    if name == "synthetic":
        if output_dim == 2:
            criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
            metric = binary_accuracy
            if canonical:
                model = CanonicalLinear(n_canonical=n_canonical, input_dimension=input_dim, num_classes=1, interpolate=interpolate).to(device)
                used_mirror = True
            else:
                model = LinearLayer(input_dim, 1).to(device)
            is_binary_classification = True
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
            metric = accuracy
            model = LinearLayer(input_dim, output_dim).to(device)
            is_binary_classification = False
    elif name == "har":
        criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
        metric = binary_accuracy
        if canonical:
            model = CanonicalLinear(n_canonical=n_canonical, input_dimension=input_dim, num_classes=1, interpolate=interpolate).to(device)
            used_mirror = True
        else:
            model = LinearLayer(input_dim, 1).to(device)
        is_binary_classification = True
    elif name == "cifar10":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        # adapt for canonical model in cifar10
        if canonical:
            model = get_canonical_mobilenet(n_classes=10, n_canonical=n_canonical,interpolate=interpolate).to(device)
            used_mirror = True
        else:
            model = get_mobilenet(n_classes=10).to(device)
        # model = get_resnet18(n_classes=10).to(device)
        is_binary_classification = False
    elif name == "cifar100":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet(n_classes=100).to(device)
        is_binary_classification = False
    elif name == "emnist" or name == "femnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN(num_classes=62).to(device)
        is_binary_classification = False
    elif name =="mnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        if canonical:
            model = CanonicalMnist(n_canonical=n_canonical,interpolate=interpolate).to(device)
            used_mirror = True
        else:
            model = MnistPerceptron().to(device)
        is_binary_classification = False
    elif name == "shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8

        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
        metric = accuracy
        model =\
            NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"]
            ).to(device)
        is_binary_classification = False
    # add covariate
    elif name == "covariate":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        if canonical:
            model = CanonicalDigitModel(n_canonical=n_canonical,interpolate=interpolate).to(device)
            used_mirror = True
        else:
            model = DigitModel().to(device)
        is_binary_classification = False
    else:
        raise NotImplementedError

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu
        )
    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    if name == "shakespeare":
        return LanguageModelingLearner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    else:
        return Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification,
            include_keyword=include_keyword,
            exclude_keyword=exclude_keyword,
            used_mirror=used_mirror
        )


def get_learners_ensemble(
        n_learners,
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None,
        include_keyword=None,
        exclude_keyword=None,
        canonical=False,
        n_canonical=0,
        interpolate=False
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param n_learners: number of learners in the ensemble
    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: LearnersEnsemble

    """
    learners = [
        get_learner(
            name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed + learner_id,
            mu=mu,
            include_keyword=include_keyword,
            exclude_keyword=exclude_keyword,
            canonical=canonical,
            n_canonical=n_canonical,
            interpolate=interpolate
        ) for learner_id in range(n_learners)
    ]

    learners_weights = torch.ones(n_learners) / n_learners
    if name == "shakespeare":
        return LanguageModelingLearnersEnsemble(learners=learners, learners_weights=learners_weights)
    else:
        return LearnersEnsemble(learners=learners, learners_weights=learners_weights)


def get_loaders(type_, root_path, batch_size, is_validation):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last

    :param type_: type of the dataset;
    :param root_path: path to the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])

    """
    # collect both train and test features and labels to reconstruct client dataset
    if type_ == "cifar10":
        inputs, targets = get_cifar10()
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
    elif type_ == "emnist":
        inputs, targets = get_emnist()
    elif type_ == "mnist":
        inputs, targets = get_mnist()
    elif type_ =="har":
        inputs, targets = get_har()
    # additional datasets for covariate
    elif type_ == "covariate":
        if "train" in root_path:
            train_iterators, val_iterators, test_iterators = get_covariate(batch_size)
            return train_iterators, val_iterators, test_iterators, OrderedDict((i, i) for i in range(5))
        else:
            return [], [], [], OrderedDict()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []
    task_idx_map = OrderedDict()  # record the order of selected tasks.

    for task_id, task_dir in enumerate(tqdm(os.listdir(root_path))):
        task_idx_map[int(task_dir.split("_")[1])] = task_id  # feat: save the task_id:client_id map to reconstruct the C matrix.
        task_data_path = os.path.join(root_path, task_dir)

        train_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=True
            )

        val_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)

    return train_iterators, val_iterators, test_iterators, task_idx_map


def get_loader(type_, path, batch_size, train, inputs=None, targets=None):
    """
    constructs a torch.utils.DataLoader object from the given path

    :param type_: type of the dataset; possible are `tabular`, `images` and `text`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return: torch.utils.DataLoader

    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "cifar10":
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets)
    elif type_ == "emnist":
        dataset = SubEMNIST(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
    elif type_ == "mnist":
        dataset = SubMNIST(path, inputs, targets)
    elif type_ == "har":
        dataset = SubHAR(path, inputs, targets)
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = ((type_ == "cifar100") or (type_ == "cifar10")) and (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)


def get_client(
        client_type,
        learners_ensemble,
        q,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally,
        client_lr,
        client_scheduler
):
    """

    :param client_type:
    :param learners_ensemble:
    :param q: fairness hyper-parameter, ony used for FFL client
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param tune_locally

    :return:

    """
    if client_type == "mixture":
        return MixtureClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
    elif client_type == "AFL":
        return AgnosticFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
    elif client_type == "FFL":
        return FFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            q=q
        )
    elif client_type == "Fedpop":
        return FedpopClient(
            learners_ensemble=learners_ensemble, 
            train_iterator=train_iterator, 
            val_iterator=val_iterator, 
            test_iterator=test_iterator, 
            logger=logger, 
            local_steps=local_steps, 
            client_lr=client_lr, 
            client_scheduler=client_scheduler,
            tune_locally=tune_locally
        )
    else:
        return Client(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )


def get_aggregator(
        aggregator_type,
        clients,
        global_learners_ensemble,
        lr,
        lr_lambda,
        mu,
        communication_probability,
        q,
        sampling_rate,
        log_freq,
        global_train_logger,
        global_test_logger,
        test_clients,
        verbose,
        seed=None,
        laplace_coef = 1e-5,
        similarity_path=None,
        saved_dir = None,
        task_id_map = None,
):
    """
    `personalized` corresponds to pFedMe

    :param aggregator_type:
    :param clients:
    :param global_learners_ensemble:
    :param lr: oly used with FLL aggregator
    :param lr_lambda: only used with Agnostic aggregator
    :param mu: penalization term, only used with L2SGD
    :param communication_probability: communication probability, only used with L2SGD
    :param q: fairness hyper-parameter, ony used for FFL client
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    if aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "personalized":
        return PersonalizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "clustered":
        return ClusteredAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "L2SGD":
        return LoopLessLocalSGDAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            communication_probability=communication_probability,
            penalty_parameter=mu,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "AFL":
        return AgnosticAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr_lambda=lr_lambda,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "FFL":
        return FFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr=lr,
            q=q,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "decentralized":
        n_clients = len(clients)
        mixing_matrix = get_mixing_matrix(n=n_clients, p=0.5, seed=seed)

        return DecentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            mixing_matrix=mixing_matrix,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == 'PartialAggregator':
        return PartialAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == 'FedpopAggregator':
        # def __init__(self, clients, global_learners_ensemble, log_freq, global_train_logger, global_test_logger, similarity_path, 
        # sampling_rate=1, sample_with_replacement=False, test_clients=None, verbose=0, 
        # seed=None, laplace_coef=1e-5, *args, **kwargs):
        if not saved_dir or not similarity_path:
            raise '''Saved_dir and similarity_path shoudn't be None in Fedpop '''
        if task_id_map is None:
            raise 'please input the task order to reorder similarity matrix!'
        return FedpopAggregator(
            clients = clients,
            global_learners_ensemble = global_learners_ensemble,
            log_freq = log_freq,
            global_train_logger = global_train_logger,
            global_test_logger = global_test_logger,
            test_clients=test_clients,
            similarity_path=similarity_path,
            saved_dir = saved_dir,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            laplace_coef=laplace_coef,
            task_id_map=task_id_map
        )
    else:
        raise NotImplementedError(
            "{aggregator_type} is not a possible aggregator type."
            " Available are: `no_communication`, `centralized`,"
            " `personalized`, `clustered`, `fednova`, `AFL`,"
            " `FFL` and `decentralized`."
        )
