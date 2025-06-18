# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch 
from typing import List, Optional
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import LambdaLR

from pfl import torch_utils
from .server_optimizers import SGD, Adam

def get_server_optimizer(server_optimizer, server_model, server_lr, server_momentum):
    if server_optimizer.lower() == 'sgd':
        return SGD(
            server_model.server_state_dict().values(), lr=server_lr, momentum=server_momentum
        )
    elif server_optimizer.lower() == 'adam':
        return Adam(server_model.server_state_dict().values(), lr=server_lr)
    else:
        raise ValueError(f'Unknown Optimizer: {server_optimizer}')

def get_client_optimizer(client_optimizer, model, num_training_steps, optimizer_args, parameters_to_choose='all'):
    # optimizer_args: client_lr, client_momentum, scheduler, lr_decay_factor, lr_decay_every, warmup_fraction
    # parameters_to_choose: accept one of ['all', 'server', 'client']
    if parameters_to_choose == 'all':
        params = model.parameters()
    elif parameters_to_choose == 'server':
        params = model.server_parameters()
    elif parameters_to_choose == 'client':
        params = model.client_parameters()
    else:
        raise ValueError(f'Unknown params_to_choose: {parameters_to_choose}')
    if client_optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=optimizer_args.client_lr, 
                                    momentum=optimizer_args.client_momentum)
    elif client_optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=optimizer_args.client_lr)
    elif client_optimizer.lower() == "md":
        # adding mirror descent for cross entropy loss.
        optimizer = MD(params, lr=optimizer_args.client_lr)
    else:
        raise ValueError(f'Unknown optimizer: {client_optimizer}')
    # Setup scheduler
    if optimizer_args.scheduler == 'const':
        lr_lambda = lambda current_step: 1.0  # mult. factor = 1.0
    elif optimizer_args.scheduler == 'linear':
        num_warmup_steps = optimizer_args.warmup_fraction * num_training_steps
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return current_step / max(1.0, num_warmup_steps)
            return max(0.0, 
                (num_training_steps - current_step) / 
                max(1.0, num_training_steps - num_warmup_steps)
            )
    elif optimizer_args.scheduler == 'expo':
        def lr_lambda(current_step):
            return min(1.0, max(0.0, optimizer_args.lr_decay_factor)) ** (current_step / num_training_steps)
    elif optimizer_args.scheduler == 'const_and_cut':
        def lr_lambda(current_step):
            factor = current_step // optimizer_args.lr_decay_every
            return optimizer_args.lr_decay_factor ** factor
    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler



class MD(Optimizer):
    """
    Mirror descent optimizer for entropy distance generation function.
    """

    def __init__(self, params, lr=required, *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False):
                 
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))


        defaults = dict(lr=lr, maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        super(MD, self).__init__(params, defaults)
    

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)
    
    @torch.no_grad()
    def step(self, penalty=None, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    
                    if penalty is not None:
                        p.grad.add_(penalty)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]

                    
            md(params_with_grad,
                d_p_list,
                lr=group['lr'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])


def md(params: List[Tensor],
        d_p_list: List[Tensor],
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        lr: float,
        maximize: bool
        ):
    if foreach is None:
    # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False
    
    func = __single_tensor_md

    return func(params,
                d_p_list,
                lr=lr,
                has_sparse_grad=has_sparse_grad,
                maximize=maximize)

def __single_tensor_md(params: List[Tensor], 
                       d_p_list: List[Tensor], 
                       lr:float, 
                       maximize: bool,
                       has_sparse_grad: bool):
    
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        d_p.data = torch.exp(-lr * d_p.data)

        unproj = param.data * torch.exp(d_p.data)
        
        # param.add_(d_p, alpha=-lr)
        param.data = unproj / torch.sum(unproj)