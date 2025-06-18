# Modified by Hao Di on 2025-06-17
# Based on original work licensed under the Apache License, Version 2.0

import os
import time
import random
import pickle

from abc import ABC, abstractmethod
from copy import deepcopy
from collections import OrderedDict as ODict
from typing import OrderedDict

import numpy as np
import numpy.linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

from utils.torch_utils import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients: List[Client]

    test_clients: List[Client]

    global_learners_ensemble: List[Learner]

    sampling_rate: proportion of clients used at each round; default is `1.`

    sample_with_replacement: is True, client are sampled with replacement; default is False

    n_clients:

    n_learners:

    clients_weights:

    model_dim: dimension if the used model

    c_round: index of the current communication round

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    global_train_logger:

    global_test_logger:

    rng: random number generator

    np_rng: numpy random number generator

    Methods
    ----------
    __init__
    mix

    update_clients

    update_test_clients

    write_logs

    save_state

    load_state

    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            *args,
            **kwargs
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.global_learners_ensemble = global_learners_ensemble
        self.device = self.global_learners_ensemble.device

        self.log_freq = log_freq
        self.verbose = verbose
        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger

        self.model_dim = self.global_learners_ensemble.model_dim

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)
        self.n_learners = len(self.global_learners_ensemble)

        self.clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = list()

        self.c_round = 0
        self.write_logs()

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def update_test_clients(self):
        for client in self.test_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)

        for client in self.test_clients:
            client.update_sample_weights()
            client.update_learners_weights()

    def write_logs(self):
        self.update_test_clients()

        for global_logger, clients in [
            (self.global_train_logger, self.clients),
            (self.global_test_logger, self.test_clients)
        ]:
            if len(clients) == 0:
                continue
            
            # for covariate
            # clients_test_loss = [0.] * len(clients)
            # clients_test_acc = [0.] * len(clients)

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs()

                if self.verbose > 1:
                    logging.info("*" * 30)
                    logging.info(f"Client {client_id}..")

                    with np.logging.infooptions(precision=3, suppress=True):
                        logging.info("Pi: ", client.learners_weights.numpy())

                    logging.info(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|")
                    logging.info(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                # for covariate
                # clients_test_loss[client_id] = test_loss
                # clients_test_acc[client_id] = test_acc * 100

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                logging.info("+" * 30)
                logging.info("Global..")
                logging.info(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |")
                logging.info(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                # logging.info(f"Clients Test Loss: {clients_test_loss} | Clients Test Acc: {clients_test_acc}% |")
                logging.info("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        if self.verbose > 0:
            logging.info("#" * 80)

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         as `.pt` file, and `learners_weights` for each client in `self.clients` as a single numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            save_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            torch.save(learner.model.state_dict(), save_path)

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            save_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            for client_id, client in enumerate(clients):
                weights[client_id] = client.learners_ensemble.learners_weights

            np.save(save_path, weights)

    def load_state(self, dir_path):
        """
        load the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         from a `.pt` file, and `learners_weights` for each client in `self.clients` from numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            chkpts_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            learner.model.load_state_dict(torch.load(chkpts_path))

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            chkpts_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            weights = np.load(chkpts_path)

            for client_id, client in enumerate(clients):
                client.learners_ensemble.learners_weights = weights[client_id]

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        if self.sample_with_replacement:
            self.sampled_clients = \
                self.rng.choices(
                    population=self.clients,
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients = self.rng.sample(self.clients, k=self.n_clients_per_round)


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        pass



class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """
    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()  # run client.n_epochs epoch

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            average_learners(learners, learner, weights=self.clients_weights)  # update global model

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(learner.model, self.global_learners_ensemble[learner_id].model)

                # use for FedProx
                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters()
                    )


class PersonalizedAggregator(CentralizedAggregator):
    r"""
    Clients do not synchronize there models, instead they only synchronize optimizers, when needed.
    # For pFedme algorithm
    """
    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(self.global_learners_ensemble[learner_id].model.parameters())


class APFLAggregator(Aggregator):
    r"""
    Implements
        `Adaptive Personalized Federated Learning`__(https://arxiv.org/abs/2003.13461)

    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            alpha,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(APFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )
        assert self.n_learners == 2, "APFL requires two learners"

        self.alpha = alpha

    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            for _ in range(client.local_steps):
                client.step(single_batch_flag=True)

                partial_average(
                    learners=[client.learners_ensemble[1]],
                    average_learner=client.learners_ensemble[0],
                    alpha=self.alpha
                )

        average_learners(
            learners=[client.learners_ensemble[0] for client in self.clients],
            target_learner=self.global_learners_ensemble[0],
            weights=self.clients_weights
        )

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for client in self.clients:

            copy_model(client.learners_ensemble[0].model, self.global_learners_ensemble[0].model)

            if callable(getattr(client.learners_ensemble[0].optimizer, "set_initial_params", None)):
                client.learners_ensemble[0].optimizer.set_initial_params(
                    self.global_learners_ensemble[0].model.parameters()
                )


class LoopLessLocalSGDAggregator(PersonalizedAggregator):
    """
    Implements L2SGD introduced in
    'Federated Learning of a Mixture of Global and Local Models'__. (https://arxiv.org/pdf/2002.05516.pdf)


    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            communication_probability,
            penalty_parameter,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(LoopLessLocalSGDAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.communication_probability = communication_probability
        self.penalty_parameter = penalty_parameter

    @property
    def communication_probability(self):
        return self.__communication_probability

    @communication_probability.setter
    def communication_probability(self, communication_probability):
        self.__communication_probability = communication_probability

    def mix(self):
        communication_flag = self.np_rng.binomial(1, self.communication_probability, 1)

        if communication_flag:
            for learner_id, learner in enumerate(self.global_learners_ensemble):
                learners = [client.learners_ensemble[learner_id] for client in self.clients]
                average_learners(learners, learner, weights=self.clients_weights)

                partial_average(
                    learners,
                    average_learner=learner,
                    alpha=self.penalty_parameter/self.communication_probability
                )

                self.update_clients()

                self.c_round += 1

                if self.c_round % self.log_freq == 0:
                    self.write_logs()

        else:
            self.sample_clients()
            for client in self.sampled_clients:
                client.step(single_batch_flag=True)


class ClusteredAggregator(Aggregator):
    """
    Implements
     `Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints`.

     Follows implementation from https://github.com/felisat/clustered-federated-learning
    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            tol_1=0.03,  # the original is 0.4
            tol_2=0.75,  # the original is 1.6
            seed=None
    ):

        super(ClusteredAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        assert self.n_learners == 1, "ClusteredAggregator only supports single learner clients."
        assert self.sampling_rate == 1.0, f"`sampling_rate` is {sampling_rate}, should be {1.0}," \
                                          f" ClusteredAggregator only supports full clients participation."

        self.tol_1 = tol_1
        self.tol_2 = tol_2

        self.global_learners = [self.global_learners_ensemble]
        self.clusters_indices = [np.arange(len(clients)).astype("int")]
        self.n_clusters = 1

    def mix(self):
        clients_updates = np.zeros((self.n_clients, self.n_learners, self.model_dim))

        for client_id, client in enumerate(self.clients):
            clients_updates[client_id] = client.step()

        similarities = np.zeros((self.n_learners, self.n_clients, self.n_clients))

        for learner_id in range(self.n_learners):
            similarities[learner_id] = pairwise_distances(clients_updates[:, learner_id, :], metric="cosine")

        similarities = similarities.mean(axis=0)

        new_cluster_indices = []
        for indices in self.clusters_indices:
            max_update_norm = np.zeros(self.n_learners)
            mean_update_norm = np.zeros(self.n_learners)

            for learner_id in range(self.n_learners):
                max_update_norm[learner_id] = LA.norm(clients_updates[indices], axis=1).max()
                mean_update_norm[learner_id] = LA.norm(np.mean(clients_updates[indices], axis=0))

            max_update_norm = max_update_norm.mean()
            mean_update_norm = mean_update_norm.mean()

            logging.info("+" * 30)
            logging.info(f"the max_update_norm is {max_update_norm}, the mean_update_norm is {mean_update_norm}")
            logging.info("+" * 30)

            if mean_update_norm < self.tol_1 and max_update_norm > self.tol_2 and len(indices) > 2 and self.c_round > 20:
                clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete")
                clustering.fit(similarities[indices][:, indices])
                cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
                cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
                new_cluster_indices += [cluster_1, cluster_2]
            else:
                new_cluster_indices += [indices]

        self.clusters_indices = new_cluster_indices

        self.n_clusters = len(self.clusters_indices)

        self.global_learners = [deepcopy(self.clients[0].learners_ensemble) for _ in range(self.n_clusters)]

        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_clients = [self.clients[i] for i in indices]
            for learner_id in range(self.n_learners):
                average_learners(
                    learners=[client.learners_ensemble[learner_id] for client in cluster_clients],
                    target_learner=self.global_learners[cluster_id][learner_id],
                    weights=self.clients_weights[indices] / self.clients_weights[indices].sum()
                )

        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_learners = self.global_learners[cluster_id]

            for i in indices:
                for learner_id, learner in enumerate(self.clients[i].learners_ensemble):
                    copy_model(
                        target=learner.model,
                        source=cluster_learners[learner_id].model
                    )

    def update_test_clients(self):
        pass


class AgnosticAggregator(CentralizedAggregator):
    """
    Implements
     `Agnostic Federated Learning`__(https://arxiv.org/pdf/1902.00146.pdf).

    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            lr_lambda,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(AgnosticAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.lr_lambda = lr_lambda

    def mix(self):
        self.sample_clients()

        clients_losses = []
        for client in self.sampled_clients:
            client_losses = client.step()
            clients_losses.append(client_losses)

        clients_losses = torch.tensor(clients_losses)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]

            average_learners(
                learners=learners,
                target_learner=learner,
                weights=self.clients_weights,
                average_gradients=True
            )

        # update parameters
        self.global_learners_ensemble.optimizer_step()

        # update clients weights
        self.clients_weights += self.lr_lambda * clients_losses.mean(dim=1)
        self.clients_weights = simplex_projection(self.clients_weights)

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class FFLAggregator(CentralizedAggregator):
    """
    Implements q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            lr,
            q=1,
            sampling_rate=1.,
            sample_with_replacement=True,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(FFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.q = q
        self.lr = lr
        assert self.sample_with_replacement, 'FFLAggregator only support sample with replacement'

    def mix(self):
        self.sample_clients()

        hs = 0
        for client in self.sampled_clients:
            hs += client.step(lr=self.lr)

        hs /= (self.lr * len(self.sampled_clients))  # take account for the lr used inside optimizer

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients]
            average_learners(
                learners=learners,
                target_learner=learner,
                weights=hs*torch.ones(len(learners)),
                average_params=False,
                average_gradients=True
            )

        # update parameters
        self.global_learners_ensemble.optimizer_step()

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class DecentralizedAggregator(Aggregator):
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            mixing_matrix,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=True,
            test_clients=None,
            verbose=0,
            seed=None):

        super(DecentralizedAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.mixing_matrix = mixing_matrix
        assert self.sampling_rate >= 1, "partial sampling is not supported with DecentralizedAggregator"

    def update_clients(self):
        pass

    def mix(self):
        # update local models
        for client in self.clients:
            client.step()

        # mix models
        mixing_matrix = torch.tensor(
            self.mixing_matrix.copy(),
            dtype=torch.float32,
            device=self.device
        )

        for learner_id, global_learner in enumerate(self.global_learners_ensemble):
            state_dicts = [client.learners_ensemble[learner_id].model.state_dict() for client in self.clients]

            for key, param in global_learner.model.state_dict().items():
                shape_ = param.shape
                models_params = torch.zeros(self.n_clients, int(np.prod(shape_)), device=self.device)

                for ii, sd in enumerate(state_dicts):
                    models_params[ii] = sd[key].view(1, -1)

                models_params = mixing_matrix @ models_params

                for ii, sd in enumerate(state_dicts):
                    sd[key] = models_params[ii].view(shape_)

            for client_id, client in enumerate(self.clients):
                client.learners_ensemble[learner_id].model.load_state_dict(state_dicts[client_id])

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class PartialAggregator(CentralizedAggregator):
    def update_clients(self,):
        for client in self.clients:
            include_keyword = client.learners_ensemble.include_keyword
            exclude_keyword = client.learners_ensemble.exclude_keyword
            for learner_id, learner in enumerate(client.learners_ensemble):
                # copy_model(learner.model, self.global_learners_ensemble[learner_id].model)
                partial_copy_model(learner.model, self.global_learners_ensemble[learner_id].model, 
                                   include_keyword=include_keyword, exclude_keyword=exclude_keyword)


def partial_copy_model(target, source, include_keyword=None, exclude_keyword=None):
    copy_state = ODict()
    # save state dict
    for key, val in source.state_dict().items():
        # remove the bn layer
        if 'bn' in key:
            continue

        if include_keyword:
            if include_keyword in key:
                copy_state[key] = val
        elif exclude_keyword:
            if exclude_keyword not in key:
                copy_state[key] = val
        else:
            copy_state[key] = val
    target.load_state_dict(copy_state, strict=False)


class FedpopAggregator(Aggregator):
    
    def __init__(self, clients, global_learners_ensemble, log_freq, global_train_logger, global_test_logger, similarity_path, saved_dir, sampling_rate=1, sample_with_replacement=False, test_clients=None, verbose=0, seed=None, laplace_coef=1e-5, task_id_map=None,*args, **kwargs):
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.global_learners_ensemble = global_learners_ensemble
        self.device = self.global_learners_ensemble.device

        self.log_freq = log_freq
        self.verbose = verbose
        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger

        self.model_dim = self.global_learners_ensemble.model_dim

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)
        self.n_learners = len(self.global_learners_ensemble)

        self.clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = list()

        self.c_round = 0

        self.similarity = np.load(similarity_path)
        self.similarity = self.similarity[list(task_id_map.keys())]  # reorder the similarity matrix
        self.task_id_map = task_id_map

        self.laplace_coef = laplace_coef
        self.client_idx = {client:i for i, client in enumerate(self.clients)}
        self.saved_dir = saved_dir  # same as the training logger path

        self.write_logs()
    
    def get_penalty(self) -> np.ndarray:
        all_factors = self.get_all_factors()
        L = np.diag(np.sum(self.similarity, axis=1))

        penalty_matrix = self.laplace_coef * L.dot(all_factors)
        return penalty_matrix

    def get_all_factors(self):
        with torch.no_grad():
            all_factors = np.array(
                list(map(lambda client: client.canonical_factor.cpu().clone().detach().numpy(), 
                    self.clients))
            )
        return all_factors

    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()
        
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            average_learners(learners, learner, weights=None)
        
        self.update_clients()
        
        penlaty_matrix = self.get_penalty()

        for client in self.sampled_clients:
            idx = self.client_idx[client]
            penalty = penlaty_matrix[idx, :]
            penalty = torch.tensor(penalty, dtype=torch.float32).cuda()  # transform nadarray into tensor
            client.mirror_step(penalty)
        
        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()
    
    def update_clients(self):
        for client in self.clients:
            include_keyword = client.learners_ensemble.include_keyword
            exclude_keyword = client.learners_ensemble.exclude_keyword
            for learner_id, learner in enumerate(client.learners_ensemble):
                # copy_model(learner.model, self.global_learners_ensemble[learner_id].model)
                partial_copy_model(learner.model, self.global_learners_ensemble[learner_id].model, 
                                include_keyword = include_keyword, exclude_keyword=exclude_keyword)


    def write_logs(self):
        self.update_test_clients()

        for global_logger, clients in [
            (self.global_train_logger, self.clients),
            (self.global_test_logger, self.test_clients)
        ]:
            if len(clients) == 0:
                continue
            
            # for covariate
            clients_test_loss = [0.] * len(clients)
            clients_test_acc = [0.] * len(clients)

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs()

                if self.verbose > 1:
                    logging.info("*" * 30)
                    logging.info(f"Client {client_id}..")

                    with np.logging.infooptions(precision=3, suppress=True):
                        logging.info("Pi: ", client.learners_weights.numpy())

                    logging.info(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|")
                    logging.info(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                # for covariate
                # clients_test_loss[client_id] = test_loss
                # clients_test_acc[client_id] = test_acc * 100

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                logging.info("+" * 30)
                logging.info("Global..")
                logging.info(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |")
                logging.info(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                # logging.info(f"Clients Test Loss: {clients_test_loss} | Clients Test Acc: {clients_test_acc}% |")
                logging.info("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        if self.verbose > 0:
            logging.info("#" * 80)
        
        all_factor = self.get_all_factors()
        cur_path = os.path.join(self.saved_dir, f'round_{self.c_round}.npy')
        np.save(cur_path, all_factor)

        # save the task_id:client_id map
        cur_path = os.path.join(self.saved_dir, f'task_id_map.pkl')
        if not os.path.exists(cur_path):
            with open(cur_path, 'wb') as f:
                pickle.dump(self.task_id_map, f)
    
    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         as `.pt` file, and `learners_weights` for each client in `self.clients` as a single numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            save_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            torch.save(learner.model.state_dict(), save_path)

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            save_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            for client_id, client in enumerate(clients):
                weights[client_id] = client.learners_ensemble.learners_weights

            np.save(save_path, weights)

        all_factor = self.get_all_factors()
        cur_path = os.path.join(self.saved_dir, f'final.npy')
        np.save(cur_path, all_factor)