# Model.py
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
        self.encoder = models.vgg16_bn(pretrained=True)

        # Projection (mlp) network
        self.projection_mlp = ProjectionMLP(
            input_dim=1000,
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

