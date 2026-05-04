import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from typing import Tuple, List

class GlobalLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class FTB_Lite(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.freq_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return F.relu(self.bn(self.freq_conv(x)))

class TSB_Lite(nn.Module):
    def __init__(self, C_A=32, C_P=16):
        super().__init__()
        self.ftb = FTB_Lite(C_A)
        self.convA = nn.Sequential(
            nn.Conv2d(C_A, C_A, kernel_size=3, padding=1),
            nn.BatchNorm2d(C_A),
            nn.ReLU(),
            nn.Conv2d(C_A, C_A, kernel_size=(7, 1), padding=(3, 0)), # Ядро 25 -> 7
            nn.BatchNorm2d(C_A),
            nn.ReLU()
        )
        self.convP = nn.Sequential(
            GlobalLayerNorm(C_P),
            nn.Conv2d(C_P, C_P, kernel_size=(7, 1), padding=(3, 0)) # Ядро 25 -> 7
        )
        self.gate_P_to_A = nn.Conv2d(C_P, C_A, kernel_size=1)
        self.gate_A_to_P = nn.Conv2d(C_A, C_P, kernel_size=1)

    def forward(self, sa, sp):
        sa_out = self.convA(self.ftb(sa))
        sp_out = self.convP(sp)
        sa_next = sa_out * torch.tanh(self.gate_P_to_A(sp_out))
        sp_next = sp_out * torch.tanh(self.gate_A_to_P(sa_out))
        return sa_next, sp_next

class PHASEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.F = 257
        C_A, C_P = 32, 16

        self.init_A = nn.Sequential(
            nn.Conv2d(2, C_A, kernel_size=3, padding=1),
            nn.BatchNorm2d(C_A), nn.ReLU()
        )
        
        self.init_P = nn.Sequential(
            nn.Conv2d(2, C_P, kernel_size=3, padding=1)
        )

        self.tsb = TSB_Lite(C_A, C_P)

        self.out_A_conv = nn.Conv2d(C_A, 4, kernel_size=1)

        self.rnn = nn.GRU(
            input_size=self.F * 4, 
            hidden_size=128, 
            bidirectional=True, 
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.F),
            nn.Sigmoid()
        )

        self.out_P_conv = nn.Conv2d(C_P, 2, kernel_size=1)

    def forward(self, x):
        B, C, T, F_dim = x.shape

        sa = self.init_A(x)
        sp = self.init_P(x)

        sa, sp = self.tsb(sa, sp)

        mask = self.out_A_conv(sa).permute(0, 2, 1, 3).reshape(B, T, -1)
        mask, _ = self.rnn(mask)
        mask = self.fc(mask).unsqueeze(1)

        phase_raw = self.out_P_conv(sp)
        phase = phase_raw / (torch.norm(phase_raw, dim=1, keepdim=True) + 1e-8)

        mag_in = torch.norm(x, dim=1, keepdim=True)

        return (mag_in * mask) * phase