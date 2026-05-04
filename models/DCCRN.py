import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from typing import Tuple, List
    
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x_r, x_i):
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        return out_r, out_i

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_t_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_t_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x_r, x_i):
        out_r = self.conv_t_r(x_r) - self.conv_t_i(x_i)
        out_i = self.conv_t_r(x_i) + self.conv_t_i(x_r)
        return out_r, out_i

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.register_buffer('running_mean_r', torch.zeros(num_features))
        self.register_buffer('running_mean_i', torch.zeros(num_features))
        self.register_buffer('running_Vrr', torch.ones(num_features))
        self.register_buffer('running_Vii', torch.ones(num_features))
        self.register_buffer('running_Vri', torch.zeros(num_features))

        self.gamma_rr = nn.Parameter(torch.ones(num_features))
        self.gamma_ii = nn.Parameter(torch.ones(num_features))
        self.gamma_ri = nn.Parameter(torch.zeros(num_features))
        self.beta_r = nn.Parameter(torch.zeros(num_features))
        self.beta_i = nn.Parameter(torch.zeros(num_features))

    def forward(self, x_r, x_i):
        B, C, F, T = x_r.shape
        if self.training:
            mean_r = x_r.mean(dim=[0, 2, 3])
            mean_i = x_i.mean(dim=[0, 2, 3])
            self.running_mean_r.data = (1 - self.momentum) * self.running_mean_r + self.momentum * mean_r
            self.running_mean_i.data = (1 - self.momentum) * self.running_mean_i + self.momentum * mean_i
        else:
            mean_r = self.running_mean_r
            mean_i = self.running_mean_i

        x_r_c = x_r - mean_r.view(1, C, 1, 1)
        x_i_c = x_i - mean_i.view(1, C, 1, 1)

        if self.training:
            Vrr = (x_r_c ** 2).mean(dim=[0, 2, 3]) + self.eps
            Vii = (x_i_c ** 2).mean(dim=[0, 2, 3]) + self.eps
            Vri = (x_r_c * x_i_c).mean(dim=[0, 2, 3])

            self.running_Vrr.data = (1 - self.momentum) * self.running_Vrr + self.momentum * Vrr
            self.running_Vii.data = (1 - self.momentum) * self.running_Vii + self.momentum * Vii
            self.running_Vri.data = (1 - self.momentum) * self.running_Vri + self.momentum * Vri
        else:
            Vrr = self.running_Vrr
            Vii = self.running_Vii
            Vri = self.running_Vri

        det = Vrr * Vii - Vri ** 2 + self.eps
        s = torch.sqrt(det)
        t = torch.sqrt(Vrr + Vii + 2 * s + self.eps)
        inv_st = 1.0 / (t * s + self.eps)

        Wrr = ((Vii + s) * inv_st).view(1, C, 1, 1)
        Wii = ((Vrr + s) * inv_st).view(1, C, 1, 1)
        Wri = (-Vri * inv_st).view(1, C, 1, 1)

        x_r_hat = Wrr * x_r_c + Wri * x_i_c
        x_i_hat = Wri * x_r_c + Wii * x_i_c

        out_r = self.gamma_rr.view(1, C, 1, 1) * x_r_hat + self.gamma_ri.view(1, C, 1, 1) * x_i_hat + self.beta_r.view(1, C, 1, 1)
        out_i = self.gamma_ri.view(1, C, 1, 1) * x_r_hat + self.gamma_ii.view(1, C, 1, 1) * x_i_hat + self.beta_i.view(1, C, 1, 1)

        return out_r, out_i

class ComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm_r = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm_i = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x_r, x_i):
        F_rr, _ = self.lstm_r(x_r)
        F_ir, _ = self.lstm_r(x_i)
        F_ri, _ = self.lstm_i(x_r)
        F_ii, _ = self.lstm_i(x_i)

        out_r = F_rr - F_ii
        out_i = F_ri + F_ir
        return out_r, out_i

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ComplexConv2d(in_c, out_c, kernel_size=(5, 2), stride=(2, 1), padding=0)
        self.bn = ComplexBatchNorm2d(out_c)
        self.prelu_r = nn.PReLU(out_c)
        self.prelu_i = nn.PReLU(out_c)

    def forward(self, x_r, x_i):
        x_r_pad = F.pad(x_r, (1, 0, 2, 2))
        x_i_pad = F.pad(x_i, (1, 0, 2, 2))
        
        x_r, x_i = self.conv(x_r_pad, x_i_pad)
        x_r, x_i = self.bn(x_r, x_i)
        x_r = self.prelu_r(x_r)
        x_i = self.prelu_i(x_i)
        return x_r, x_i

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, is_last=False):
        super().__init__()
        self.conv_t = ComplexConvTranspose2d(
            in_c, out_c, kernel_size=(5, 2), stride=(2, 1), 
            padding=(2, 0), output_padding=(1, 0)
        )
        self.is_last = is_last
        if not is_last:
            self.bn = ComplexBatchNorm2d(out_c)
            self.prelu_r = nn.PReLU(out_c)
            self.prelu_i = nn.PReLU(out_c)

    def forward(self, x_r, x_i):
        x_r, x_i = self.conv_t(x_r, x_i)
        
        x_r = x_r[..., :-1]
        x_i = x_i[..., :-1]
        
        if not self.is_last:
            x_r, x_i = self.bn(x_r, x_i)
            x_r = self.prelu_r(x_r)
            x_i = self.prelu_i(x_i)
        return x_r, x_i

class DCCRN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [1, 32, 64, 128, 128, 256, 256]
        
        self.encoders = nn.ModuleList()
        for i in range(6):
            self.encoders.append(EncoderBlock(channels[i], channels[i+1]))

        self.lstm = ComplexLSTM(input_size=1280, hidden_size=256, num_layers=2)
        self.dense_r = nn.Linear(256, 1280)
        self.dense_i = nn.Linear(256, 1280)

        self.decoders = nn.ModuleList()
        dec_channels = [256, 256, 128, 128, 64, 32]
        for i in range(6):
            in_c = dec_channels[i] + channels[6-i]
            out_c = channels[5-i]
            is_last = (i == 5)
            self.decoders.append(DecoderBlock(in_c, out_c, is_last))

        self.register_buffer('window', torch.hann_window(400))

    def forward(self, wav):
        stft = torch.stft(wav, n_fft=512, hop_length=100, win_length=400, 
                          window=self.window, return_complex=True)
        stft_r = stft.real.unsqueeze(1)
        stft_i = stft.imag.unsqueeze(1)

        stft_r = F.pad(stft_r, (0, 0, 0, 320 - 257))
        stft_i = F.pad(stft_i, (0, 0, 0, 320 - 257))

        enc_outputs_r, enc_outputs_i = [], []
        x_r, x_i = stft_r, stft_i
        for encoder in self.encoders:
            x_r, x_i = encoder(x_r, x_i)
            enc_outputs_r.append(x_r)
            enc_outputs_i.append(x_i)

        B, C, F_dim, T = x_r.shape  # C=256, F_dim=5
        lstm_in_r = x_r.permute(0, 3, 1, 2).reshape(B, T, C * F_dim)
        lstm_in_i = x_i.permute(0, 3, 1, 2).reshape(B, T, C * F_dim)

        lstm_out_r, lstm_out_i = self.lstm(lstm_in_r, lstm_in_i)

        dense_out_r = self.dense_r(lstm_out_r) - self.dense_i(lstm_out_i)
        dense_out_i = self.dense_r(lstm_out_i) + self.dense_i(lstm_out_r)

        x_r = dense_out_r.view(B, T, C, F_dim).permute(0, 2, 3, 1)
        x_i = dense_out_i.view(B, T, C, F_dim).permute(0, 2, 3, 1)

        for i, decoder in enumerate(self.decoders):
            skip_r = enc_outputs_r[5 - i]
            skip_i = enc_outputs_i[5 - i]
            
            x_r = torch.cat([x_r, skip_r], dim=1)
            x_i = torch.cat([x_i, skip_i], dim=1)
            
            x_r, x_i = decoder(x_r, x_i)

        out_r = x_r[:, 0, :257, :]
        out_i = x_i[:, 0, :257, :]

        mask_mag = torch.tanh(torch.sqrt(out_r**2 + out_i**2 + 1e-8))
        mask_phase = torch.atan2(out_i, out_r)
        
        mask_r = mask_mag * torch.cos(mask_phase)
        mask_i = mask_mag * torch.sin(mask_phase)

        S_r = stft.real * mask_r - stft.imag * mask_i
        S_i = stft.real * mask_i + stft.imag * mask_r
        S_complex = torch.complex(S_r, S_i)

        enh_wav = torch.istft(S_complex, n_fft=512, hop_length=100, win_length=400, 
                              window=self.window, length=wav.shape[1])
        return enh_wav