import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from typing import Tuple, List
    
class SDE:
    def __init__(self, config):
        self.gamma = config.gamma
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.log_sig = math.log(self.sigma_max / self.sigma_min)

    def g(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min)**t * math.sqrt(2 * self.log_sig)

    def marginal_prob(self, x0, y, t):
        t = t.view(-1, 1, 1, 1)
        mean = torch.exp(-self.gamma * t) * x0 + (1 - torch.exp(-self.gamma * t)) * y
        
        factor = 2 * self.gamma + 2 * self.log_sig
        var = (self.sigma_min**2 / factor) * (
            (self.sigma_max / self.sigma_min)**(2*t) - torch.exp(-2 * self.gamma * t)
        )
        return mean, torch.sqrt(var)

def compress_stft(c, alpha=0.5):
    mag = torch.abs(c)
    phase = torch.angle(c)
    return (mag ** alpha) * torch.exp(1j * phase)

def decompress_stft(c_tilde, alpha=0.5):
    mag = torch.abs(c_tilde)
    phase = torch.angle(c_tilde)
    return (mag ** (1.0 / alpha)) * torch.exp(1j * phase)

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        xr, xi = x.real, x.imag
        real_out = self.conv_r(xr) - self.conv_i(xi)
        imag_out = self.conv_r(xi) + self.conv_i(xr)
        return torch.complex(real_out, imag_out)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.conv_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        xr, xi = x.real, x.imag
        real_out = self.conv_r(xr) - self.conv_i(xi)
        imag_out = self.conv_r(xi) + self.conv_i(xr)
        return torch.complex(real_out, imag_out)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, x):
        xr, xi = x.real, x.imag
        real_out = self.fc_r(xr) - self.fc_i(xi)
        imag_out = self.fc_r(xi) + self.fc_i(xr)
        return torch.complex(real_out, imag_out)

class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return torch.complex(self.bn_r(x.real), self.bn_i(x.imag))

class RealGaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim=128, scale=30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        x = t[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    
class StandardBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, is_decoder=False, output_padding=0):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        if is_decoder:
            self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding=padding, output_padding=output_padding)
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding=padding)

        self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()

        self.time_embed = nn.Sequential(
            nn.Linear(128, out_c),
            nn.ReLU()
        )

    def forward(self, x, t_emb):
        h = self.conv(x)
        h = self.norm(h)
        
        t_embed = self.time_embed(t_emb)
        t_embed = t_embed.view(t_embed.shape[0], t_embed.shape[1], 1, 1)
        
        h = h + t_embed
        return self.act(h)

class SGMSE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.t_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU()
        )

        self.enc1 = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.down2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())

        self.final = nn.Conv2d(16, 2, 3, padding=1)

    def forward(self, x_t, t, y):
        t_emb = self.t_proj(t.unsqueeze(-1)).view(-1, 64, 1, 1)

        x = torch.cat([x_t.real, x_t.imag, y.real, y.imag], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))

        b = self.bottleneck(self.down2(e2))
        b = b + t_emb 

        def pad_match(target, source):
            diff_f = source.shape[2] - target.shape[2]
            diff_t = source.shape[3] - target.shape[3]
            return F.pad(target, [diff_t // 2, diff_t - diff_t // 2, diff_f // 2, diff_f - diff_f // 2])

        d2 = self.up2(b)
        d2 = pad_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = pad_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)
        out = pad_match(out, x_t)

        return torch.complex(out[:, 0:1, :, :], out[:, 1:2, :, :])