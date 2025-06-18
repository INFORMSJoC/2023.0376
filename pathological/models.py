# Modified by Hao Di on 2025-06-17
# Based on original work licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch import Tensor

from typing import Optional, List, Callable

from collections import OrderedDict


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)

class CanonicalLinear(nn.Module):
    def __init__(self, n_canonical, input_dimension, num_classes, bias=True, interpolate=False) -> None:
        super(CanonicalLinear, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc_list = nn.ModuleList([nn.Linear(input_dimension, num_classes) for _ in range(n_canonical)])
        self.canonical_factor = nn.parameter.Parameter(torch.tensor([1./n_canonical]*n_canonical, dtype=torch.float32), 
                                                  requires_grad=True)
        self.interpolate = interpolate
        self.n_canonical = n_canonical
    
    def forward(self, x):
        if self.interpolate:
            output = self.linear_interpolate(x, self.fc_list)
        else:
            x = torch.cat([fc(x).unsqueeze_(-1) for fc in self.fc_list], dim=-1)
            output = x @ self.canonical_factor
        return output

    def linear_interpolate(self, x, fc_list):
        """
        implement the interpolation in linear functions
        """
        basic_cnt = []  # count the number of parameters in each module of a basic model
        param_iter = [model.parameters() for model in fc_list]
        for m in fc_list[0].modules():
            basic_cnt.append(len(m.state_dict()))
        for i in range(len(basic_cnt)):
            cnt = 0
            param_list = []
            while cnt < basic_cnt[i]:
                params = [next(iter) for iter in param_iter]
                cnt += 1
                param_size = params[0].shape
                n = len(param_size)
                cat_param = torch.cat(params, dim=0).view(self.n_canonical, *(param_size))
                cat_param = cat_param.permute(*list(range(1, n+1)), 0)
                w = torch.matmul(cat_param, self.canonical_factor)
                param_list.append(w)
            x = nn.functional.linear(x, *param_list)
        return x

class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.classifier(x)
        return x

class CanonicalDigitModel(nn.Module):
    def __init__(self, n_canonical, interpolate=False, num_classes=10) -> None:
        super(CanonicalDigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        # self.fc2 = nn.Linear(2048, 512)
        self.fc2 = nn.ModuleList([nn.Linear(2048, 512) for _ in range(n_canonical)])
        self.bn5 = nn.BatchNorm1d(512)
        # self.fc3 = nn.Linear(512, num_classes)
        self.fc3 = nn.ModuleList([nn.Linear(512, num_classes) for _ in range(n_canonical)])

        
        # self.fc = nn.ModuleList([nn.Linear(28*28, 128) for _ in range(n_canonical)])
        # self.classifier = nn.ModuleList([nn.Linear(128, 10) for _ in range(n_canonical)])
        self.interpolate = interpolate
        self.n_canonical = n_canonical

        self.canonical_factor = nn.parameter.Parameter(torch.tensor([1./n_canonical]*n_canonical, dtype=torch.float32), 
                                                  requires_grad=True)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)

        if self.interpolate:
            x = self.linear_interpolate(x, self.fc2)
        else:
            x = torch.cat([fc(x).unsqueeze_(-1) for fc in self.fc2], dim=-1)
            x = x @ self.canonical_factor
        x = self.bn5(x)
        x = F.relu(x)


        if self.interpolate:
            x = self.linear_interpolate(x, self.fc3)
        else:
            x = torch.cat([fc(x).unsqueeze_(-1) for fc in self.fc3], dim=-1)
            x = x @ self.canonical_factor
        return x

    def linear_interpolate(self, x, fc_list):
        """
        implement the interpolation in linear functions
        """
        basic_cnt = []  # count the number of parameters in each module of a basic model
        param_iter = [model.parameters() for model in fc_list]
        for m in fc_list[0].modules():
            basic_cnt.append(len(m.state_dict()))
        for i in range(len(basic_cnt)):
            cnt = 0
            param_list = []
            while cnt < basic_cnt[i]:
                params = [next(iter) for iter in param_iter]
                cnt += 1
                param_size = params[0].shape
                n = len(param_size)
                cat_param = torch.cat(params, dim=0).view(self.n_canonical, *(param_size))
                cat_param = cat_param.permute(*list(range(1, n+1)), 0)
                w = torch.matmul(cat_param, self.canonical_factor)
                param_list.append(w)
            x = nn.functional.linear(x, *param_list)
        return x

class MnistPerceptron(nn.Module):
    def __init__(self) -> None:
        super(MnistPerceptron, self).__init__()
        self.fc = nn.Linear(28*28, 128)
        self.classifier = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc(x))
        output = self.classifier(x)
        return output

class CanonicalMnist(nn.Module):
    def __init__(self, n_canonical, interpolate=False) -> None:
        super(CanonicalMnist, self).__init__()
        
        self.fc = nn.ModuleList([nn.Linear(28*28, 128) for _ in range(n_canonical)])
        self.classifier = nn.ModuleList([nn.Linear(128, 10) for _ in range(n_canonical)])
        self.interpolate = interpolate
        self.n_canonical = n_canonical

        self.canonical_factor = nn.parameter.Parameter(torch.tensor([1./n_canonical]*n_canonical, dtype=torch.float32), 
                                                  requires_grad=True)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        if self.interpolate:
            x = self.linear_interpolate(x, self.fc)
        else:
            x = torch.cat([fc(x).unsqueeze_(-1) for fc in self.fc], dim=-1)
            x = x @ self.canonical_factor
        
        x = F.relu(x)
        
        if self.interpolate:
            out = self.linear_interpolate(x, self.classifier)
        else:
            x = torch.cat([fc(x).unsqueeze_(-1) for fc in self.classifier], dim=-1)
            out = x @ self.canonical_factor
        return out

    def linear_interpolate(self, x, fc_list):
        """
        implement the interpolation in linear functions
        """
        basic_cnt = []  # count the number of parameters in each module of a basic model
        param_iter = [model.parameters() for model in fc_list]
        for m in fc_list[0].modules():
            basic_cnt.append(len(m.state_dict()))
        for i in range(len(basic_cnt)):
            cnt = 0
            param_list = []
            while cnt < basic_cnt[i]:
                params = [next(iter) for iter in param_iter]
                cnt += 1
                param_size = params[0].shape
                n = len(param_size)
                cat_param = torch.cat(params, dim=0).view(self.n_canonical, *(param_size))
                cat_param = cat_param.permute(*list(range(1, n+1)), 0)
                w = torch.matmul(cat_param, self.canonical_factor)
                param_list.append(w)
            x = nn.functional.linear(x, *param_list)
        return x

class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output


def get_vgg11(n_classes):
    """
    creates VGG11 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

    return model


def get_squeezenet(n_classes):
    """
    creates SqueezeNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes

    return model


def get_mobilenet(n_classes):
    """
    creates MobileNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model


def get_resnet18(n_classes):
    """
    creates Resnet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_resnet34(n_classes):
    """
    creates Resnet34 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_canonical_mobilenet(n_classes, n_canonical,interpolate=False):
    original_model = models.mobilenet_v2(pretrained=True)
    state_dict = original_model.state_dict()
    loaded_dict = OrderedDict()

    for key in state_dict:
        if 'classifier' not in key:
            loaded_dict[key] = state_dict[key]
        
    model = CanonicalMobilenet(n_canonical=n_canonical,num_classes=n_classes,interpolate=interpolate)
    model.load_state_dict(loaded_dict, strict=False)
    return model


class CanonicalMobilenet(models.MobileNetV2):
    def __init__(self, 
                 n_canonical,
                 num_classes: int = 1000, 
                 width_mult: float = 1, 
                 inverted_residual_setting: Optional[List[List[int]]] = None, 
                 round_nearest: int = 8, 
                 block: Optional[Callable[..., nn.Module]] = None, 
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 interpolate=False) -> None:
        super().__init__(num_classes, width_mult, inverted_residual_setting, round_nearest, block, norm_layer)

        self.n_canonical = n_canonical
        self.interpolate = interpolate

        self.last_out = nn.Dropout(0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        self.classifier = nn.ModuleList([nn.Linear(self.last_channel, num_classes) for _ in range(n_canonical)])
        self.canonical_factor = nn.parameter.Parameter(torch.tensor([1./n_canonical]*n_canonical, dtype=torch.float32), 
                                                  requires_grad=True)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.last_out(x)

        if not self.interpolate:
            x = torch.cat([fc(x).unsqueeze_(-1) for fc in self.classifier], dim=-1)
            x = x @ self.canonical_factor
        else:
            x = self.linear_interpolate(x, self.classifier)
        return x

    def linear_interpolate(self, x, fc_list):
        """
        implement the interpolation in linear functions
        """
        basic_cnt = []  # count the number of parameters in each module of a basic model
        param_iter = [model.parameters() for model in fc_list]
        for m in fc_list[0].modules():
            basic_cnt.append(len(m.state_dict()))
        for i in range(len(basic_cnt)):
            cnt = 0
            param_list = []
            while cnt < basic_cnt[i]:
                params = [next(iter) for iter in param_iter]
                cnt += 1
                param_size = params[0].shape
                n = len(param_size)
                cat_param = torch.cat(params, dim=0).view(self.n_canonical, *(param_size))
                cat_param = cat_param.permute(*list(range(1, n+1)), 0)
                w = torch.matmul(cat_param, self.canonical_factor)
                param_list.append(w)
            x = nn.functional.linear(x, *param_list)
        return x