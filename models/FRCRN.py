import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from typing import Tuple, List

class ComplexBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)
    def forward(self, r, i):
        return self.bn_r(r), self.bn_i(i)

class ComplexConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=(1,1), padding=(0,0)):
        super().__init__()
        self.conv_r = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.conv_i = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
    def forward(self, r, i):
        return self.conv_r(r) - self.conv_i(i), self.conv_r(i) + self.conv_i(r)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=(1,1), padding=(0,0), output_padding=(0,0)):
        super().__init__()
        self.conv_r = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding, output_padding, bias=False)
        self.conv_i = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding, output_padding, bias=False)
    def forward(self, r, i):
        return self.conv_r(r) - self.conv_i(i), self.conv_r(i) + self.conv_i(r)

class CFSMN(nn.Module):
    def __init__(self, channels, memory_size=20, axis='freq'):
        super().__init__()
        self.axis = axis
        self.memory_size = memory_size
        self.conv_r = nn.Conv1d(channels, channels, memory_size + 1, groups=channels, bias=False)
        self.conv_i = nn.Conv1d(channels, channels, memory_size + 1, groups=channels, bias=False)

    def forward(self, r, i):
        B, C, F_dim, T_dim = r.shape
        if self.axis == 'freq':
            r_in = r.permute(0, 3, 1, 2).reshape(B * T_dim, C, F_dim)
            i_in = i.permute(0, 3, 1, 2).reshape(B * T_dim, C, F_dim)
        else:
            r_in = r.permute(0, 2, 1, 3).reshape(B * F_dim, C, T_dim)
            i_in = i.permute(0, 2, 1, 3).reshape(B * F_dim, C, T_dim)

        r_pad = F.pad(r_in, (self.memory_size, 0))
        i_pad = F.pad(i_in, (self.memory_size, 0))

        out_r = self.conv_r(r_pad) - self.conv_i(i_pad)
        out_i = self.conv_r(i_pad) + self.conv_i(r_pad)

        res_r, res_i = r_in + out_r, i_in + out_i

        if self.axis == 'freq':
            return res_r.view(B, T_dim, C, F_dim).permute(0, 2, 3, 1), res_i.view(B, T_dim, C, F_dim).permute(0, 2, 3, 1)
        return res_r.view(B, F_dim, C, T_dim).permute(0, 2, 1, 3), res_i.view(B, F_dim, C, T_dim).permute(0, 2, 1, 3)

class CCBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        inter_channels = max(1, channels // 8)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, r, i):
        mag = torch.sqrt(r**2 + i**2 + 1e-8)
        ca = self.channel_attn(mag)
        r, i, mag = r * ca, i * ca, mag * ca
        avg_m = torch.mean(mag, dim=1, keepdim=True)
        max_m = torch.max(mag, dim=1, keepdim=True)[0]
        sa = self.spatial_attn(torch.cat([avg_m, max_m], dim=1))
        return r * sa, i * sa
    
class FRCRN(nn.Module):
    def __init__(self, n_fft, hop_len, win_len):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop_len, win_len
        self.register_buffer('window', torch.hann_window(self.win))

        self.encoders = nn.ModuleList([
            self._make_enc_block(1 if k==0 else 128) for k in range(6)
        ])

        self.bottleneck = nn.Sequential(
            CFSMN(128, 20, 'time'),
            CFSMN(128, 20, 'time')
        )

        self.decoders = nn.ModuleList([
            self._make_dec_block(
                in_c=256,
                out_c=(128 if k < 5 else 1),
                out_pad=(0, 0),
                attn_c=128
            ) for k in range(6)
        ])

        self.alpha = nn.Parameter(torch.tensor([0.95]))

    def _make_enc_block(self, in_c):
        return nn.ModuleDict({
            'conv': ComplexConv2d(in_c, 128, (5, 2), stride=(2, 1), padding=(2, 0)),
            'bn': ComplexBatchNorm(128),
            'fsmn': CFSMN(128, 20, 'freq')
        })

    def _make_dec_block(self, in_c, out_c, out_pad, attn_c):
        return nn.ModuleDict({
            'attn': CCBAM(attn_c),
            'conv': ComplexConvTranspose2d(in_c, out_c, (5, 2), stride=(2, 1), padding=(2, 0), output_padding=out_pad),
            'bn': ComplexBatchNorm(out_c),
            'fsmn': CFSMN(out_c, 20, 'freq')
        })

    def forward(self, x):
        B, _, L = x.shape
        stft = torch.stft(x.squeeze(1), self.n_fft, self.hop, self.win, self.window,
                          return_complex=True, center=True)
        noisy_r, noisy_i = stft.real.unsqueeze(1), stft.imag.unsqueeze(1)

        r, i = noisy_r, noisy_i
        skips = []

        for enc in self.encoders:
            r = F.pad(r, (1, 0, 0, 0))
            i = F.pad(i, (1, 0, 0, 0))

            r, i = enc['conv'](r, i)
            r, i = r[:, :, :, :-1], i[:, :, :, :-1]

            r, i = enc['bn'](r, i)
            r, i = F.leaky_relu(r, 0.2), F.leaky_relu(i, 0.2)
            r, i = enc['fsmn'](r, i)

            skips.append((r, i))

        r, i = self.bottleneck[0](r, i)
        r, i = self.bottleneck[1](r, i)

        for idx, dec in enumerate(self.decoders):
            skip_r, skip_i = skips[-(idx+1)]
            sr, si = dec['attn'](skip_r, skip_i)

            if r.shape[2:] != sr.shape[2:]:
                r = F.interpolate(r, size=sr.shape[2:], mode='bilinear', align_corners=True)
                i = F.interpolate(i, size=si.shape[2:], mode='bilinear', align_corners=True)

            r, i = torch.cat([r, sr], 1), torch.cat([i, si], 1)

            r = F.pad(r, (1, 0, 0, 0))
            i = F.pad(i, (1, 0, 0, 0))
            r, i = dec['conv'](r, i)
            r, i = r[:, :, :, :-1], i[:, :, :, :-1]

            r, i = dec['bn'](r, i)
            if idx < 5:
                r, i = F.leaky_relu(r, 0.2), F.leaky_relu(i, 0.2)
                r, i = dec['fsmn'](r, i)

        m_r, m_i = torch.tanh(r), torch.tanh(i)
        est_r = noisy_r * m_r - noisy_i * m_i
        est_i = noisy_r * m_i + noisy_i * m_r

        est_wav = torch.istft(torch.complex(est_r.squeeze(1), est_i.squeeze(1)),
                              self.n_fft, self.hop, self.win, self.window,
                              length=L, center=True).unsqueeze(1)

        return self.alpha * est_wav + (1 - self.alpha) * x