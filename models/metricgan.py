import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from typing import Tuple, List
    
class LearnableSigmoid(nn.Module):
    def __init__(self, in_features: int = 257, beta: float = 1.2):
        super(LearnableSigmoid, self).__init__()
        self.beta = beta
        self.alpha = nn.Parameter(torch.ones(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.alpha * x)

class metricgan(nn.Module):
    def __init__(self, input_dim: int = 257, hidden_dim: int = 200, num_layers: int = 2):
        super(metricgan, self).__init__()
        self.blstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 300)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(300, 257)
        self.learnable_sigmoid = LearnableSigmoid(in_features=257, beta=1.2)
        self.mask_lower_bound = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.blstm(x)
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        mask = self.learnable_sigmoid(out)
        mask = torch.clamp(mask, min=self.mask_lower_bound)
        return x * mask

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.conv1 = spectral_norm(nn.Conv2d(2, 15, (5, 5)))
        self.conv2 = spectral_norm(nn.Conv2d(15, 15, (5, 5)))
        self.conv3 = spectral_norm(nn.Conv2d(15, 15, (5, 5)))
        self.conv4 = spectral_norm(nn.Conv2d(15, 15, (5, 5)))
        self.fc1 = spectral_norm(nn.Linear(15, 50))
        self.fc2 = spectral_norm(nn.Linear(50, 10))
        self.fc3 = spectral_norm(nn.Linear(10, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3: x = x.unsqueeze(1)
        if y.dim() == 3: y = y.unsqueeze(1)
        inp = torch.cat([x, y], dim=1)
        out = self.leaky_relu(self.conv1(inp))
        out = self.leaky_relu(self.conv2(out))
        out = self.leaky_relu(self.conv3(out))
        out = self.leaky_relu(self.conv4(out))
        out = torch.mean(out, dim=(2, 3))
        out = self.leaky_relu(self.fc1(out))
        out = self.leaky_relu(self.fc2(out))
        return self.fc3(out)