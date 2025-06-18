from collections import OrderedDict
from typing import Optional
import torch
import torch.nn as nn
from torchinfo import summary

from .base_model import PFLBaseModel


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

GROUP_NORM_LOOKUP = {
    16: 2,  # -> channels per group: 8
    32: 4,  # -> channels per group: 8
    64: 8,  # -> channels per group: 8
    128: 8,  # -> channels per group: 16
    256: 16,  # -> channels per group: 16
    512: 32,  # -> channels per group: 16
    1024: 32,  # -> channels per group: 32
    2048: 32,  # -> channels per group: 64
}

def create_group_norm(num_channels):
    return torch.nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels)

class ResidualBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None) -> None:
        super(ResidualBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = create_group_norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = create_group_norm(planes)
        self.downsample = downsample
        self.stride = stride

        # self.use_adapter = False
        # self.adapter1 = False
        # self.adapter2 = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        # if self.use_adapter:
        #     out = out + self.adapter1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # if self.use_adapter:
        #     out = out + self.adapter2(out)
        out = self.bn2(out)
        if self.downsample is not None:  # reshape x
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    # def add_adapters(self, dropout=0.):
    #     if not self.use_adapter:
    #         self.use_adapter = True
    #         self.adapter1 = AdapterBlock(self.planes, dropout)
    #         self.adapter2 = AdapterBlock(self.planes, dropout)

# class AdapterBlock(nn.Module):
#     def __init__(self, planes, dropout):
#         super().__init__()
#         # self.bn = nn.BatchNorm2d(planes)
#         self.bn = create_group_norm(planes)
#         self.conv = conv1x1(planes, planes)  # 1x1 convolution
#         self.dropout = nn.Dropout(dropout)
#         # initialize
#         nn.init.normal_(self.conv.weight, 0, 1e-4)
#         # nn.init.constant_(self.conv.bias, 0.0)  # no bias

#     def forward(self, x):
#         identity = x
#         out = self.bn(x)  # Batch norm
#         out = self.conv(self.dropout(out))  # 1x1 conv
#         out += identity  # skip connection
#         return out

class ResNetGN(PFLBaseModel):
    def __init__(self, n_canonical, layers=(2, 2, 2, 2), num_classes=62, original_size=False, interpolate=False):
        # if original_size: expect (3, 224, 224) images, else expect (1, 28, 28)
        assert n_canonical >= 1
        super().__init__()
        self.inplanes = 64
        self.drop_i = nn.Dropout(0.)
        self.drop_o = nn.Dropout(0.)
        if original_size:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = torch.nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  # 28 * 28
        self.bn1 = create_group_norm(self.inplanes)  # num_groups = 8
        self.relu = nn.ReLU(inplace=True)
        if original_size:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_list = nn.ModuleList([nn.Linear(512, num_classes) for _ in range(n_canonical)])
        self.canonical_factor = nn.parameter.Parameter(torch.tensor([1./n_canonical]*n_canonical, dtype=torch.float32), 
                                                  requires_grad=True)
        self.interpolate = interpolate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.is_on_client = None
        self.is_on_server = None

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                # nn.BatchNorm2d(planes),
                create_group_norm(planes),
            )
        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.drop_i(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
    
        # x = torch.cat([fc(self.drop_o(x)).unsqueeze_(-1) for fc in self.fc_list], dim=-1)
        # x = x @ self.canonical_factor
        x = self.drop_o(x)
        if not self.interpolate:
            # mix predictions
            x = torch.cat([fc(x).unsqueeze_(-1) for fc in self.fc_list], dim=-1)  # mix predictions
            x = x @ self.canonical_factor
        else:
            # parameters interpolation
            x = self.linear_interpolation(x, self.fc_list)
        return x 

    def linear_interpolation(self, x, fc_list):
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

    def print_summary(self, train_batch_size):
        raise NotImplementedError

    def split_server_and_client_params(self, client_mode, layers_to_client, adapter_hidden_dim=-1, dropout=0.):
        device = next(self.parameters()).device
        if self.is_on_client is not None:
            raise ValueError('This model has already been split across clients and server.')
        assert client_mode in ['none', 'res_layer', 'inp_layer', 'out_layer', 'adapter', 'interpolate', 'finetune', "canonical"] 
        # Prepare
        if layers_to_client is None:  # no layers to client
            layers_to_client = []
        if client_mode == 'res_layer' and len(layers_to_client) is None:
            raise ValueError(f'No residual blocks to finetune. Nothing to do')
        is_on_server = None
        
        # Set requires_grad based on `train_mode`
        if client_mode in ['none', None]:
            # no parameters on the client
            def is_on_client(name):
                return False
        elif 'res_layer' in client_mode:
            # Specific residual blocks are sent to client (available layers are [1, 2, 3, 4])
            def is_on_client(name):
                return any([f'layer{i}' in name for i in layers_to_client])
        elif client_mode in ['inp_layer']:
            # First convolutional layer is sent to client
            def is_on_client(name):
                return (name in ['conv1.weight', 'bn1.weight', 'bn1.bias'])  # first conv + bn
            self.drop_i = nn.Dropout(dropout)
        elif client_mode in ['out_layer']:
            # Final linear layer is sent to client
            def is_on_client(name):
                return (name in ['fc.weight', 'fc.bias'])  # final fc
            self.drop_o = nn.Dropout(dropout)
        elif client_mode in ['adapter']:
            # Train adapter modules (+ batch norm)
            def is_on_client(name):
                return ('adapter' in name) or ('bn1' in name) or ('bn2' in name)
            # Add adapter modules
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer.children():
                    # each block is of type `ResidualBlock`
                    block.add_adapters(dropout)
        elif client_mode in ["canonical"]:
            def is_on_client(name):
                return "canonical" in name
            self.drop_o = nn.Dropout(dropout)
        elif client_mode == 'interpolate':  # both on client and server
            is_on_client = lambda _: True
            is_on_server = lambda _: True
        elif client_mode == 'finetune':  # all on client
            is_on_client = lambda _: True
            is_on_server = lambda _: False
        else:
            raise ValueError(f'Unknown client_mode: {client_mode}')
        if is_on_server is None:
            def is_on_server(name): 
                return not is_on_client(name)
        
        self.is_on_client = is_on_client
        self.is_on_server = is_on_server
        self.to(device)


class Canonical_EmnistResNetGN(ResNetGN):
    def __init__(self, n_canonical, interpolate):
        super().__init__(n_canonical, layers=(2, 2, 2, 2), num_classes=62, original_size=False, interpolate=interpolate)
        self.n_canonical = n_canonical

    def print_summary(self, train_batch_size):
        device = next(self.parameters()).device
        print(summary(self, input_size=(train_batch_size, 1, 28, 28), device=device))
    
