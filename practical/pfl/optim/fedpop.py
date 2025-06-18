# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from collections import OrderedDict
import copy
import random
import numpy as np
import pandas as pd
import torch

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .pfl_base import PartialPFLBase
from .utils import get_client_optimizer, MD

from pfl import torch_utils, utils




class FedPop(PartialPFLBase):
    """
    Implementation for heterogenous populated federated learning. 
    train local params until the global params have been updated.
    """
    def __init__(self, train_fed_loader, available_clients, clients_to_cache, server_model, 
                 server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm,
                 save_dir, seed, similarity_path, laplace_coef, save_client_params_to_disk=False, stateless_clients=False,
                 client_var_l2_reg_coef=0.0, client_var_prox_to_init=False, max_num_pfl_updates=1000, **kwargs):
        
        client_model = copy.deepcopy(server_model)
        # load the similarity matrix
        # TODO:if each_neighbor_simpling is positive, load the sparse matrix of scipy
        self.is_sparse = kwargs['is_sparse']
        if self.is_sparse:
            # TODO: using sketching if available clients is relatively large.
            self.sparse_matrix = sparse.load_npz(kwargs['sparse_path'])
            id2idx = {client_id:idx for idx, client_id in enumerate(train_fed_loader.available_clients)}
            self.available_idx = [id2idx[client_id] for client_id in available_clients]
            similary_matrix = self.__class__.rejust_cosine(self.sparse_matrix[self.available_idx, :])
            self.similary_matrix = pd.DataFrame(similary_matrix, index=available_clients, columns=available_clients)
        else:
            self.similary_matrix = pd.read_csv(similarity_path, index_col=0)

        self.laplace_coef = laplace_coef

        super().__init__(
            train_fed_loader, available_clients, clients_to_cache, server_model, client_model, 
            server_optimizer, server_lr, server_momentum, max_grad_norm, clip_grad_norm, save_dir, seed,
            save_client_params_to_disk, stateless_clients, client_var_l2_reg_coef, client_var_prox_to_init, 
            max_num_pfl_updates)

        
    
    def get_client_laplance_penlaty(self, client_id:List[str]):
        # TODO: adding the if - else structure for the sampling neighbors in stackoverflow.
        L = np.diag(np.sum(self.similary_matrix, axis=1))
        with torch.no_grad():
            # client_param shape is (K, )
            # saved_client_params Dict(client_id: Dict(name: params))
            client_param_npval = np.array(
                list(map(lambda x: x["canonical_factor"].cpu().clone().detach().numpy(), 
                self.saved_client_params.values()))
            )
        
        penalty_matrix = pd.DataFrame(self.laplace_coef * L.dot(client_param_npval), index=self.similary_matrix.index)
        return penalty_matrix.loc[client_id].values
    
        
    def run_one_round(self, num_clients_per_round, num_local_epochs, client_optimizer, client_optimizer_args):
        server_losses = []
        client_losses = []
        client_deltas = []
        num_data_per_client = []

        sampled_clients = self.sample_clients(num_clients_per_round)
        clients_penalty = self.get_client_laplance_penlaty(sampled_clients)

        # train the server parameters in each client
        for i, client_id in enumerate(sampled_clients):
            # load the client model
            self.load_client_model(client_id)
            # update the combined model to the correct mix of client parameter and server parameter
            self.reset_combined_model()

            # load the client dataset
            client_loader = self.train_fed_loader.get_client_dataloader(client_id)
            self.combined_model.train()

            avg_loss1, num_data = self.run_local_updates(
                client_loader, num_local_epochs, client_optimizer, client_optimizer_args
            )

            server_losses.append(avg_loss1)
            num_data_per_client.append(num_data)

            # obtain the difference of server parameters between combined model and server model.
            client_grad = self.get_client_grad()
            client_deltas.append(client_grad)
        
        # combine local updates to update the server model
        combined_grad = torch_utils.weighted_average_of_state_dicts(  # state dict
            client_deltas, num_data_per_client
        )
        self.server_optimizer.step(combined_grad)

        # train the client parameters in each client
        for i, client_id in enumerate(sampled_clients):
            self.load_client_model(client_id)
            self.reset_combined_model()

            tensor_penalty = torch.tensor(clients_penalty[i], dtype=torch.float32)

            client_loader = self.train_fed_loader.get_client_dataloader(client_id)
            self.combined_model.train()


            avg_loss2, _ = self.run_local_updates(
                client_loader, num_local_epochs, "md", client_optimizer_args, 
                parameters_choose = "client", laplace_penalty=tensor_penalty,
            )

            client_losses.append(avg_loss2)
            
            self.update_local_model()
            self.save_client_model(client_id)

        total_loss = [(loss1 + loss2)/2 for loss1, loss2 in zip(server_losses, client_losses)]

        return np.average(total_loss, weights=num_data_per_client)
    
    def get_client_grad(self):
        old_server_params = self.server_model.server_state_dict()
        new_server_params = self.combined_model.server_state_dict()
        server_param_grad = OrderedDict((k, old_server_params[k] - new_server_params[k]) for k in old_server_params.keys())
        return server_param_grad
    
    def update_local_model(self):
        new_client_params = self.combined_model.client_state_dict()
        self.client_model.load_state_dict(new_client_params, strict=False)

    def run_local_updates(
            self, client_loader, num_local_epochs,
            client_optimizer_name:str, client_optimizer_args,  parameters_choose="server", laplace_penalty=None):
        total_num_local_steps = num_local_epochs * len(client_loader)
        use_regularization = False
        use_early_stopping = False

        # Optimize client parameters first
        if parameters_choose == "server":
            self.combined_model.client_params_requires_grad_(False)
            self.combined_model.server_params_requires_grad_(True)

            client_optimizer, client_scheduler = get_client_optimizer(
                client_optimizer_name, self.combined_model, total_num_local_steps, 
                client_optimizer_args, parameters_to_choose=parameters_choose,
            )
        elif parameters_choose == "client":
            if laplace_penalty is None:
                raise ValueError("there should a params-shaped laplace penalty for md method!")
            self.combined_model.client_params_requires_grad_(True)
            self.combined_model.server_params_requires_grad_(False)

            use_regularization = True
            use_early_stopping = True
            client_optimizer_name = "md"
            client_optimizer, client_scheduler = get_client_optimizer(
                client_optimizer_name, self.combined_model, total_num_local_steps, 
                client_optimizer_args, parameters_to_choose="client"
            )
        elif parameters_choose == "all":
            self.combined_model.client_params_requires_grad_(True)
            self.combined_model.server_params_requires_grad_(True)

            client_optimizer, client_scheduler = get_client_optimizer(
                client_optimizer_name, self.combined_model, total_num_local_steps, client_optimizer_args, 
                parameters_to_choose=parameters_choose,
            )
        else:
            raise ValueError("the chosen parameter should be one of [client, server, all]")
        
        avg_loss, num_data = self.local_update_helper(
            client_loader, num_local_epochs, client_optimizer, client_scheduler,
            use_regularization=use_regularization, use_early_stopping=use_early_stopping, 
            laplace_penalty=laplace_penalty
        )
        return avg_loss, num_data
    
    def local_update_helper(
        self, client_loader, num_local_epochs, client_optimizer, client_scheduler, 
        use_regularization=False, use_early_stopping=False, laplace_penalty=None,
    ):
        device = next(self.combined_model.parameters()).device
        if laplace_penalty is not None:
            if not isinstance(laplace_penalty, torch.Tensor):
                laplace_penalty = torch.tensor(laplace_penalty, dtype=torch.float32)
            laplace_penalty = laplace_penalty.to(device)
        count = 0
        avg_loss = 0.0
        for _ in range(num_local_epochs):
            for x, y in client_loader:
                x, y = x.to(device), y.to(device)
                client_optimizer.zero_grad()
                yhat = self.combined_model(x)
                loss = self.loss_fn(yhat, y)
                if use_regularization:
                    loss = loss + self.get_client_l2_penalty()
                avg_loss = avg_loss * count / (count + 1) + loss.item() / (count + 1) 
                count += 1
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.combined_model.parameters(), self.max_grad_norm)
                if laplace_penalty is not None:
                    # adapt for mirror descent
                    if isinstance(client_optimizer, MD): 
                        client_optimizer.step(laplace_penalty)
                    else:
                        raise ValueError("the penalty should be none if the optimize isn't MD.")
                else:
                    client_optimizer.step()
                client_scheduler.step()
                if use_early_stopping and count >= self.max_num_pfl_updates:
                    break
        return avg_loss, len(client_loader)
    


    def finetune_one_client(
        self, client_loader, num_local_epochs, 
        client_optimizer_name, client_optimizer_args, laplace_penalty):
        
        total_num_local_steps = num_local_epochs * len(client_loader)

        # Optimize client parameters only
        self.combined_model.client_params_requires_grad_(True)
        self.combined_model.server_params_requires_grad_(False)

        client_optimizer_name = "md"
        client_optimizer, client_scheduler = get_client_optimizer(
            client_optimizer_name, self.combined_model, total_num_local_steps, 
            client_optimizer_args, parameters_to_choose="client",)

        avg_loss, num_data = self.local_update_helper(
            client_loader, num_local_epochs, client_optimizer, client_scheduler,
            use_regularization=True, use_early_stopping=True, laplace_penalty=laplace_penalty
        )
        return avg_loss, num_data
    

    def finetune_all_clients(self, num_local_epochs, client_optimizer, client_optimizer_args):
        # return loss, is_updated
        if self.client_model is None:  # skip finetuning if no client model
            return 0.0, False
        client_losses = []
        num_data_per_client = []
        client_penalty = self.get_client_laplance_penlaty(list(self.clients_to_cache))

        # Run local training on each client only on cached clients
        for i, client_id in enumerate(self.clients_to_cache):
            # load client model 
            self.load_client_model(client_id)
            # update combined model to be the correct mix of local and global models and set it to train mode
            self.reset_combined_model() 
            self.combined_model.train()
            # load client laplace penalty
            penalty_ = client_penalty[i]
            # run local updates
            client_loader = self.train_fed_loader.get_client_dataloader(client_id)

            avg_loss, num_data = self.finetune_one_client(
                client_loader, num_local_epochs, client_optimizer, 
                client_optimizer_args, penalty_
            )

            client_losses.append(avg_loss)
            num_data_per_client.append(num_data)

            # update local model
            self.update_local_model()  # state_dict w/ server params 

            # save updated client_model
            self.save_client_model(client_id)
        return np.average(client_losses, weights=num_data_per_client), True


    @staticmethod
    def rejust_cosine(matrix:sparse.coo_matrix) -> np.ndarray:
        res = cosine_similarity(matrix)
        return 0.5 + 0.5 * res

    '''
    # TODO:sekecting
    def penalty_from_sparse(self, client_ids:List[int]):
        penalties = []
        for select_id in client_ids:
            select_neighbor = self.select_neighbors(select_id)  # random select neighbors
            client_penalty = self.sketching(select_id, select_neighbor)  # compute penalty for select_id
            penalties.append(client_penalty)
        return penalties

    def select_neighbors(self, select_id:str) -> List[str]:
        candidates_set = set(random.choices(self.a), k=self.num_neighbor)
        while len(candidates_set) < 0.5 * self.num_neighbor:
            new_candidates = set(random.choices(self.a), k=self.num_neighbor)
            candidates_set |= new_candidates
        if select_id not in candidates_set:
            candidates_set.add(select_id)
        return list(candidates_set)
    
    # def sketching(self, select_id, select_neighbors):
    '''