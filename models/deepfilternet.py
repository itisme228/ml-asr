import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from typing import Tuple, List
    
class ERBFilterBank(nn.Module):
    def __init__(self, n_fft=512, sr=16000, n_erb=32):
        super().__init__()
        self.n_fft = n_fft
        self.n_erb = n_erb
        freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
        erb_freqs = self._hz_to_erb(freqs)
        
        erb_centers = np.linspace(erb_freqs[0], erb_freqs[-1], n_erb)
        fb = np.zeros((n_erb, n_fft // 2 + 1))
        for i in range(n_erb):
            lower = erb_centers[i-1] if i > 0 else erb_freqs[0]
            upper = erb_centers[i+1] if i < n_erb - 1 else erb_freqs[-1]
            center = erb_centers[i]
            
            fb[i] = np.interp(freqs, 
                              self._erb_to_hz(np.array([lower, center, upper])), 
                              np.array([0, 1, 0]))
            
        self.register_buffer("fb", torch.FloatTensor(fb))

    def _hz_to_erb(self, hz): return 21.4 * np.log10(1 + hz / 229.0)
    def _erb_to_hz(self, erb): return 229.0 * (10**(erb / 21.4) - 1)

    def forward(self, spec_mag):
        return torch.matmul(self.fb, spec_mag)

class DeepFilteringLayer(nn.Module):
    def __init__(self, order=5):
        super().__init__()
        self.order = order

    def forward(self, X, coefs):
        B, _, freqs, T = X.shape
        X_padded = F.pad(X, (self.order - 1, 0))
        X_unfolded = X_padded.unfold(3, self.order, 1) 
        C_re = coefs[..., 0]
        C_im = coefs[..., 1]
        
        X_re = X_unfolded[:, 0]
        X_im = X_unfolded[:, 1]
        
        out_re = torch.sum(X_re * C_re - X_im * C_im, dim=-1)
        out_im = torch.sum(X_re * C_im + X_im * C_re, dim=-1)
        
        return torch.stack([out_re, out_im], dim=1)
    
class GLinear(nn.Module):
    def __init__(self, in_f, out_f, groups=4):
        super().__init__()
        self.op = nn.Conv1d(in_f, out_f, kernel_size=1, groups=groups)

    def forward(self, x):
        return self.op(x.transpose(1, 2)).transpose(1, 2)

class DF2_Encoder(nn.Module):
    def __init__(self, erb_dim=32, freq_bins=257, hidden_dim=256):
        super().__init__()
        self.in_conv_erb = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), 0),
            nn.Conv2d(1, 16, kernel_size=(3, 3))
        )
        self.conv1x3_erb = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))

        self.in_conv_cplx = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), 0),
            nn.Conv2d(2, 16, kernel_size=(3, 3))
        )
        self.conv1x3_cplx = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))

        self.glinear = GLinear(32 * (erb_dim + freq_bins), hidden_dim, groups=4)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, erb, cplx):
        e = F.elu(self.in_conv_erb(erb.transpose(2, 3)))
        e = F.elu(self.conv1x3_erb(e))
        B, C, F_e, T = e.shape
        e = e.permute(0, 3, 1, 2).reshape(B, T, -1)

        c = F.elu(self.in_conv_cplx(cplx.transpose(2, 3)))
        c = F.elu(self.conv1x3_cplx(c))
        B, C, F_c, T = c.shape
        c = c.permute(0, 3, 1, 2).reshape(B, T, -1)

        x = torch.cat([e, c], dim=-1)

        x = F.elu(self.glinear(x))
        out, _ = self.gru(x)
        return out

class DF2_ERB_Decoder(nn.Module):
    def __init__(self, erb_dim=32, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.pconv = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1), groups=1)
        self.glinear = GLinear(hidden_dim, hidden_dim, groups=4)
        self.out = nn.Linear(hidden_dim, erb_dim)

    def forward(self, x):
        B, T, D = x.shape
        g, _ = self.gru(x)
        p = self.pconv(g.unsqueeze(1)).squeeze(1)
        x = F.elu(self.glinear(p))
        gains = torch.sigmoid(self.out(x))
        return gains, p

class DF2_DF_Decoder(nn.Module):
    def __init__(self, freq_bins=257, hidden_dim=256, order=5):
        super().__init__()
        self.order = order
        self.freq_bins = freq_bins
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = GLinear(hidden_dim, freq_bins * order * 2, groups=1)

    def forward(self, enc_out, pconv_out):
        x = enc_out + pconv_out
        x, _ = self.gru(x)
        x = self.out(x)
        
        B, T, _ = x.shape
        x = x.view(B, T, self.freq_bins, self.order, 2).transpose(1, 2)
        return x
    
class DeepFilterNet2(nn.Module):
    def __init__(self, n_fft=512, sr=16000, n_erb=32):
        super().__init__()
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1
        self.order = 5
        
        self.erb_bank = ERBFilterBank(n_fft, sr, n_erb)
        self.encoder = DF2_Encoder(n_erb, self.freq_bins)
        self.erb_dec = DF2_ERB_Decoder(n_erb)
        self.df_dec = DF2_DF_Decoder(self.freq_bins, order=self.order)
        self.df_layer = DeepFilteringLayer(order=self.order)

    def forward(self, X_complex):
        mag = torch.abs(X_complex)
        erb_feat = self.erb_bank(mag).log1p()
        
        erb_in = erb_feat.permute(0, 2, 1).unsqueeze(1)
        cplx_in = torch.stack([X_complex.real, X_complex.imag], dim=1).transpose(2, 3)
        
        enc_out = self.encoder(erb_in, cplx_in)
        
        erb_gains, pconv_out = self.erb_dec(enc_out)
        
        lin_gains = torch.matmul(self.erb_bank.fb.T, erb_gains.transpose(1, 2))
        X_stage1 = X_complex * lin_gains
        
        df_coefs = self.df_dec(enc_out, pconv_out)
        
        X_s1_split = torch.stack([X_stage1.real, X_stage1.imag], dim=1)
        
        S_hat_split = self.df_layer(X_s1_split, df_coefs)
        
        return torch.complex(S_hat_split[:, 0], S_hat_split[:, 1])