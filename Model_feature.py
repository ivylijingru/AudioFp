import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class PredictorMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    
cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, feature, num_class=10):
        super().__init__()
        self.feature = feature
        self.classifier = nn.Sequential(
            nn.Linear(4 * 12 * 512, 512),
            # nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        output = self.feature(x)
        # print('output', output.shape)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 1
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)] # stride默认为1,即保持图像尺寸不变
        if batch_norm == True:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers) # *list能提取列表中的元素

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))
def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))
def vgg16_bn():
    return VGG(make_layers(cfg['C'], batch_norm=True))
def vgg19_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

class SimSiam(nn.Module):

    def __init__(
        self,
#         backbone: str,
        latent_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
    ) -> None:

        super().__init__()

        # Encoder network
        self.encoder = vgg11_bn()

        # Projection (mlp) network
        self.projection_mlp = ProjectionMLP(
            input_dim=256,
            hidden_dim=proj_hidden_dim,
            output_dim=latent_dim
        )

        # Predictor network (h)
        self.predictor_mlp = PredictorMLP(
            input_dim=latent_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=latent_dim
        )

    def forward(self, x: torch.Tensor):
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, e: torch.Tensor) -> torch.Tensor:
        return self.projection_mlp(e)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor_mlp(z)

