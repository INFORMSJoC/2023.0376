# Modified by Hao Di on 2025-06-17
# Based on original work licensed under the Apache License, Version 2.0

import torch

class Learner:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner

    criterion (torch.nn.modules.loss): loss function used to train the `model`, should have reduction="none"

    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar

    device (str or torch.device):

    optimizer (torch.optim.Optimizer):

    lr_scheduler (torch.optim.lr_scheduler):

    is_binary_classification (bool): whether to cast labels to float or not, if `BCELoss`
    is used as criterion this should be set to True

    Methods
    ------
    compute_gradients_and_loss:

    optimizer_step: perform one optimizer step, requires the gradients to be already computed.

    fit_batch: perform an optimizer step over one batch

    fit_epoch:

    fit_batches: perform successive optimizer steps over successive batches

    fit_epochs:

    evaluate_iterator: evaluate `model` on an iterator

    gather_losses:

    get_param_tensor: get `model` parameters as a unique flattened tensor

    free_memory: free the memory allocated by the model weights

    free_gradients:
    """

    def __init__(
            self, model,
            criterion,
            metric,
            device,
            optimizer,
            lr_scheduler=None,
            is_binary_classification=False,
            include_keyword=None,
            exclude_keyword=None,
            used_mirror = False
    ):

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_binary_classification = is_binary_classification

        # introduce partial shared model
        self.include_keyword = include_keyword  # for partial FL, 'classifier'
        self.exclude_keyword = exclude_keyword  # for proposed method, 'canonical'

        self.model_dim = int(self.get_param_tensor().shape[0])
        self.used_mirror = used_mirror
    
    def server_parameters(self):
        return [param for (name, param) in self.model.named_parameters() if 'canonical' not in name]

    def canonical_parameters(self):
        return [param for (name, param) in self.model.named_parameters() if 'canonical' in name]
    
    def server_params_requires_grad(self, requires_grad):
        for param in self.server_parameters():
            param.requires_grad_(requires_grad)

    def canonical_params_requires_grad(self, requires_grad):
        for param in self.canonical_parameters():
            param.requires_grad_(requires_grad)
        

    def optimizer_step(self):
        """
         perform one optimizer step, requires the gradients to be already computed.
        """
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        self.model.train()

        # if self.used_mirror:
        #     self.server_params_requires_grad(True)
        #     self.canonical_params_requires_grad(False)

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)

            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)  # the weight vector of each sample
        else:
            loss = loss_vec.mean()

        loss.backward()

        return loss.detach()

    def fit_batch(self, batch, weights=None):
        """
        perform an optimizer step over one batch drawn from `iterator`

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()

        if self.used_mirror:
            self.server_params_requires_grad(True)
            self.canonical_params_requires_grad(False)

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y) / len(y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.detach(), metric.detach()

    def fit_epoch(self, iterator, weights=None):
        """
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()

        if self.used_mirror:
            self.server_params_requires_grad(True)
            self.canonical_params_requires_grad(False)

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y)
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()
            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).detach()

        return global_loss / n_samples, global_metric / n_samples

    def gather_losses(self, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)

                y_pred = self.model(x)
                all_losses[indices] = self.criterion(y_pred, y).squeeze()

        return all_losses

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()
        
        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, _ in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            with torch.no_grad():
                y_pred = self.model(x)

                global_loss += self.criterion(y_pred, y).sum().detach()
                global_metric += self.metric(y_pred, y).detach()

            n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples

    def fit_epochs(self, iterator, n_epochs, weights=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch(iterator, weights)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """
        
        param_list = []

        # for param in self.model.parameters():
        #     param_list.append(param.data.view(-1, ))
        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                if self.include_keyword:
                    if self.include_keyword in key:
                        param_list.append(param.data.view(-1, ))
                elif self.exclude_keyword:
                    if self.exclude_keyword not in key:
                        param_list.append(param.data.view(-1, ))
                else:
                    param_list.append(param.data.view(-1, ))
        return torch.cat(param_list)

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor

        :return: torch.tensor

        """
        grad_list = []

        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         grad_list.append(param.grad.data.view(-1, ))
        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                if param.grad is None:
                    continue
                if self.include_keyword:
                    if self.include_keyword in key:
                        grad_list.append(param.grad.data.view(-1, ))
                elif self.exclude_keyword:
                    if self.exclude_keyword not in key:
                        grad_list.append(param.grad.data.view(-1, ))
                else:
                    grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def free_memory(self):
        """
        free the memory allocated by the model weights

        """
        del self.optimizer
        del self.model

    def free_gradients(self):
        """
        free memory allocated by gradients

        """
        self.optimizer.zero_grad(set_to_none=True)

    def mirror_step(self, iterator, n_epochs, optimizer, scheduler, penalty=None):
        self.model.train()
        
        assert self.used_mirror == True

        self.server_params_requires_grad(False)
        self.canonical_params_requires_grad(True)


        def one_step():
            global_loss = 0.
            global_metric = 0.
            n_samples = 0

            for x, y, indices in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                n_samples += y.size(0)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)
                optimizer.zero_grad()

                y_pred = self.model(x)
                loss_vec = self.criterion(y_pred, y)
                loss = loss_vec.mean()

                loss.backward()

                optimizer.step(penalty)

                global_loss += loss.detach() * loss_vec.size(0)
                global_metric += self.metric(y_pred, y).detach()
            return global_loss / n_samples, global_metric / n_samples

        for epoch in range(n_epochs):
            one_step()
            if scheduler:
                scheduler.step()
        return 



class LanguageModelingLearner(Learner):
    def fit_epoch(self, iterator, weights=None):

        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device)
            y = y.to(self.device)

            n_samples += y.size(0)

            chunk_len = y.size(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)
            loss_vec = self.criterion(y_pred, y)

            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
            else:
                loss = loss_vec.mean()

            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0) / chunk_len
            global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def fit_batch(self, batch, weights=None):

        self.model.train()

        x, y, indices = batch
        x = x.to(self.device)
        y = y.to(self.device)

        n_samples = y.size(0)
        chunk_len = y.size(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        self.optimizer.step()

        global_loss = loss.detach() * loss_vec.size(0) / chunk_len
        global_metric = self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        raise NotImplementedError

    def gather_losses(self, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        n_samples = len(iterator.dataset)
        predictions = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)
                predictions[indices] = self.criterion(y_pred, y).mean(axis=1)

        return predictions

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y, _ in iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                n_samples += y.size(0)

                chunk_len = y.size(1)

                y_pred = self.model(x)
                global_loss += self.criterion(y_pred, y).sum().detach() / chunk_len
                global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples