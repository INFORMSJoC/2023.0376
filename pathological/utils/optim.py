# Modified by Hao Di on 2025-06-17
# Based on original work licensed under the Apache License, Version 2.0

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import numpy as np

from typing import Optional, List
from torch import Tensor


class ProxSGD(Optimizer):
    r"""Adaptation of  torch.optim.SGD to proximal stochastic gradient descent (optionally with momentum),
     presented in `Federated optimization in heterogeneous networks`__.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Attributes
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    mu (float, optional): parameter for proximal SGD
    momentum (float, optional): momentum factor (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)

    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = ProxSGD(model.parameters(), lr=0.1, mu=0.01,momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, mu=0., momentum=0., dampening=0.,
                 weight_decay=0., nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ProxSGD, self).__init__(params, defaults)

        self.mu = mu

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['initial_params'] = torch.clone(p.data)

    def __setstate__(self, state):
        super(ProxSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # add proximal term
                d_p.add_(p.data - param_state['initial_params'], alpha=self.mu)

                p.data.add_(d_p, alpha=-group['lr'])

        return loss

    def set_initial_params(self, initial_params):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        initial_param_groups = list(initial_params)
        if len(initial_param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(initial_param_groups[0], dict):
            initial_param_groups = [{'params': initial_param_groups}]

        for param_group, initial_param_group in zip(self.param_groups, initial_param_groups):
            for param, initial_param in zip(param_group['params'], initial_param_group['params']):
                param_state = self.state[param]
                param_state['initial_params'] = torch.clone(initial_param.data)


def get_optimizer(optimizer_name, model, lr_initial, mu=0.):
    """
    Gets torch.optim.Optimizer given an optimizer name, a model and learning rate

    :param optimizer_name: possible are adam and sgd
    :type optimizer_name: str
    :param model: model to be optimized
    :type optimizer_name: nn.Module
    :param lr_initial: initial learning used to build the optimizer
    :type lr_initial: float
    :param mu: proximal term weight; default=0.
    :type mu: float
    :return: torch.optim.Optimizer

    """

    if optimizer_name == "adam":
        return optim.Adam(
            [param for name, param in model.named_parameters() if param.requires_grad and 'canonical' not in name],
            lr=lr_initial,
            weight_decay=5e-4
        )

    elif optimizer_name == "sgd":
        return optim.SGD(
            [param for name, param in model.named_parameters() if param.requires_grad and 'canonical' not in name],  # non-canonical parameters
            lr=lr_initial,
            momentum=0.9,
            weight_decay=5e-4
        )

    elif optimizer_name == "prox_sgd":
        return ProxSGD(
            [param for name, param in model.named_parameters() if param.requires_grad and 'canonical' not in name],
            mu=mu,
            lr=lr_initial,
            momentum=0.,
            weight_decay=5e-4
        )
    else:
        raise NotImplementedError("Other optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer

    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler

    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")



class MD(Optimizer):
    """
    Mirror descent optimizer for entropy distance generation function.
    """

    def __init__(self, params, lr=required, weight_decay=0, *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False):
                 
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, maximize=maximize, foreach=foreach,
                        weight_decay = weight_decay,
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
                weight_decay=group['weight_decay'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])


def md(params: List[Tensor],
        d_p_list: List[Tensor],
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        lr: float,
        maximize: bool,
        weight_decay: float,
        ):
    if foreach is None:
    # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False
    
    func = __single_tensor_md

    return func(params,
                d_p_list,
                lr=lr,
                has_sparse_grad=has_sparse_grad,
                maximize=maximize,
                weight_decay = weight_decay)

def __single_tensor_md(params: List[Tensor], 
                       d_p_list: List[Tensor], 
                       lr:float, 
                       maximize: bool,
                       weight_decay: float,
                       has_sparse_grad: bool):
    
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        d_p.data = torch.exp(-lr * d_p.data)

        unproj = param.data * torch.exp(d_p.data)
        
        # param.add_(d_p, alpha=-lr)
        param.data = unproj / torch.sum(unproj)