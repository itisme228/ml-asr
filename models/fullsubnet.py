import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from typing import Tuple, List
    
class FullSubNet(nn.Module):
    def __init__(self, num_freqs=257, N=15):
        super(FullSubNet, self).__init__()
        self.F = num_freqs
        self.N = N
        
        self.full_lstm = nn.LSTM(input_size=self.F, hidden_size=512, num_layers=2, batch_first=True)
        self.full_linear = nn.Linear(512, self.F)
        self.full_relu = nn.ReLU()

        self.sub_lstm = nn.LSTM(input_size=2*N + 2, hidden_size=384, num_layers=2, batch_first=True)
        self.sub_linear = nn.Linear(384, 2)

    def forward(self, x_mag):

        x_mag = x_mag.transpose(1, 2)
        B, T, F_dim = x_mag.shape

        mu_full = x_mag.mean(dim=[1, 2], keepdim=True)
        x_mag_norm = x_mag / (mu_full + 1e-8)

        full_out, _ = self.full_lstm(x_mag_norm)
        g_full = self.full_relu(self.full_linear(full_out))

        x_padded = F.pad(x_mag.transpose(1, 2), (0, 0, self.N, self.N), mode='circular')

        x_unfolded = x_padded.unfold(1, 2*self.N + 1, 1)

        g_full_expanded = g_full.transpose(1, 2).unsqueeze(-1)
        sub_input = torch.cat([x_unfolded, g_full_expanded], dim=-1)

        mu_sub = sub_input.mean(dim=2, keepdim=True)
        sub_input_norm = sub_input / (mu_sub + 1e-8)

        sub_input_reshaped = sub_input_norm.reshape(B * F_dim, T, -1)

        sub_out, _ = self.sub_lstm(sub_input_reshaped)
        cirm = self.sub_linear(sub_out)
        
        return cirm.view(B, F_dim, T, 2).transpose(1, 2)