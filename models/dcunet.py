import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from typing import Tuple, List

def init_unitary_complex(shape):
    fan_in = shape[1] * shape[2] * shape[3]
    fan_out = shape[0] * shape[2] * shape[3]
    scale = math.sqrt(2.0 / (fan_in + fan_out))
    
    re_w = np.random.normal(size=(shape[0], fan_in))
    im_w = np.random.normal(size=(shape[0], fan_in))
    x = re_w + 1j * im_w
    
    if shape[0] < fan_in:
        q, _ = np.linalg.qr(x.T)
        q = q.T
    else:
        q, _ = np.linalg.qr(x)
        
    q *= scale
    return torch.from_numpy(q.real).float(), torch.from_numpy(q.imag).float()

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight_r = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.weight_i = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        
        re_init, im_init = init_unitary_complex(self.weight_r.shape)
        self.weight_r.data.copy_(re_init.view_as(self.weight_r))
        self.weight_i.data.copy_(im_init.view_as(self.weight_i))

    def forward(self, x):
        x_r, x_i = x.real, x.imag
        out_r = F.conv2d(x_r, self.weight_r, stride=self.stride, padding=self.padding) - \
                F.conv2d(x_i, self.weight_i, stride=self.stride, padding=self.padding)
        out_i = F.conv2d(x_r, self.weight_i, stride=self.stride, padding=self.padding) + \
                F.conv2d(x_i, self.weight_r, stride=self.stride, padding=self.padding)
        return torch.complex(out_r, out_i)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.kernel_size = kernel_size
        
        self.weight_r = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        self.weight_i = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))

        re_init, im_init = init_unitary_complex((in_channels, out_channels, *kernel_size))
        self.weight_r.data.copy_(re_init.view_as(self.weight_r))
        self.weight_i.data.copy_(im_init.view_as(self.weight_i))

    def forward(self, x, output_size=None):
        x_r, x_i = x.real, x.imag
        
        out_padding = (0, 0)
        if output_size is not None:
            h_out_base = (x_r.shape[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
            w_out_base = (x_r.shape[3] - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
            
            out_padding = (
                max(0, output_size[2] - h_out_base),
                max(0, output_size[3] - w_out_base)
            )

        out_r = F.conv_transpose2d(x_r, self.weight_r, stride=self.stride, padding=self.padding, output_padding=out_padding) - \
                F.conv_transpose2d(x_i, self.weight_i, stride=self.stride, padding=self.padding, output_padding=out_padding)
        out_i = F.conv_transpose2d(x_r, self.weight_i, stride=self.stride, padding=self.padding, output_padding=out_padding) + \
                F.conv_transpose2d(x_i, self.weight_r, stride=self.stride, padding=self.padding, output_padding=out_padding)

        return torch.complex(out_r, out_i)

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma_rr = nn.Parameter(torch.ones(num_features, 1, 1))
        self.gamma_ii = nn.Parameter(torch.ones(num_features, 1, 1))
        self.gamma_ri = nn.Parameter(torch.zeros(num_features, 1, 1))
        self.beta_r = nn.Parameter(torch.zeros(num_features, 1, 1))
        self.beta_i = nn.Parameter(torch.zeros(num_features, 1, 1))
        self.register_buffer('run_mu_r', torch.zeros(num_features))
        self.register_buffer('run_mu_i', torch.zeros(num_features))
        self.register_buffer('run_Vrr', torch.ones(num_features))
        self.register_buffer('run_Vii', torch.ones(num_features))
        self.register_buffer('run_Vri', torch.zeros(num_features))

    def forward(self, x):
        x_r, x_i = x.real, x.imag
        if self.training:
            mu_r = x_r.mean(dim=(0, 2, 3), keepdim=True)
            mu_i = x_i.mean(dim=(0, 2, 3), keepdim=True)
            x_r_c, x_i_c = x_r - mu_r, x_i - mu_i
            Vrr = (x_r_c ** 2).mean(dim=(0, 2, 3))
            Vii = (x_i_c ** 2).mean(dim=(0, 2, 3))
            Vri = (x_r_c * x_i_c).mean(dim=(0, 2, 3))
            with torch.no_grad():
                self.run_mu_r = (1 - self.momentum) * self.run_mu_r + self.momentum * mu_r.squeeze()
                self.run_mu_i = (1 - self.momentum) * self.run_mu_i + self.momentum * mu_i.squeeze()
                self.run_Vrr = (1 - self.momentum) * self.run_Vrr + self.momentum * Vrr
                self.run_Vii = (1 - self.momentum) * self.run_Vii + self.momentum * Vii
                self.run_Vri = (1 - self.momentum) * self.run_Vri + self.momentum * Vri
        else:
            mu_r, mu_i = self.run_mu_r.view(1, -1, 1, 1), self.run_mu_i.view(1, -1, 1, 1)
            Vrr, Vii, Vri = self.run_Vrr, self.run_Vii, self.run_Vri
            x_r_c, x_i_c = x_r - mu_r, x_i - mu_i

        det = Vrr * Vii - Vri ** 2
        s = torch.sqrt(det + self.eps)
        t = torch.sqrt(Vrr + Vii + 2 * s + self.eps)
        Wrr, Wii, Wri = (Vii + s) / (t * s), (Vrr + s) / (t * s), -Vri / (t * s)
        Wrr, Wii, Wri = Wrr.view(1, -1, 1, 1), Wii.view(1, -1, 1, 1), Wri.view(1, -1, 1, 1)
        
        x_r_w = Wrr * x_r_c + Wri * x_i_c
        x_i_w = Wri * x_r_c + Wii * x_i_c
        out_r = self.gamma_rr * x_r_w + self.gamma_ri * x_i_w + self.beta_r
        out_i = self.gamma_ri * x_r_w + self.gamma_ii * x_i_w + self.beta_i
        return torch.complex(out_r, out_i)

class ComplexLeakyReLU(nn.Module):
    def forward(self, x):
        return torch.complex(F.leaky_relu(x.real, 0.2), F.leaky_relu(x.imag, 0.2))

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.conv = ComplexConv2d(in_ch, out_ch, kernel_size, stride, padding=(kernel_size[0]//2, kernel_size[1]//2))
        self.bn = ComplexBatchNorm2d(out_ch)
        self.act = ComplexLeakyReLU()

    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, is_last=False):
        super().__init__()
        self.trans_conv = ComplexConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding=(kernel_size[0]//2, kernel_size[1]//2))
        self.is_last = is_last
        if not is_last:
            self.bn = ComplexBatchNorm2d(out_ch)
            self.act = ComplexLeakyReLU()

    def forward(self, x, skip_target):
        x = self.trans_conv(x, output_size=skip_target.size())
        return x if self.is_last else self.act(self.bn(x))

class LargeDCUnet20(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [45] + [90] * 9  
        kernels = [(7, 5), (7, 5)] + [(5, 3)] * 8
        strides = [(2, 2)] * 5 + [(2, 1)] * 4 + [(2, 1)]
        
        self.encoders = nn.ModuleList()
        in_c = 1
        for out_c, k, s in zip(channels, kernels, strides):
            self.encoders.append(EncoderBlock(in_c, out_c, k, s))
            in_c = out_c
            
        self.decoders = nn.ModuleList()
        dec_channels, dec_kernels, dec_strides = channels[::-1], kernels[::-1], strides[::-1]
        for i in range(10):
            in_c = channels[-1] if i == 0 else dec_channels[i] * 2 
            out_c = dec_channels[i + 1] if i < 9 else 1
            self.decoders.append(DecoderBlock(in_c, out_c, dec_kernels[i], dec_strides[i], is_last=(i == 9)))

    def forward(self, x):
        skips = []
        out = x
        for enc in self.encoders:
            out = enc(out)
            skips.append(out)

        rev_skips = skips[::-1]
        targets = skips[:-1][::-1] + [x] 
        
        out = rev_skips[0]

        for i, dec in enumerate(self.decoders):
            if i == 0:
                out = dec(out, targets[i])
            else:
                out = torch.cat([out, rev_skips[i]], dim=1)
                out = dec(out, targets[i])
                
        return out

class DCUNET(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.dcunet = LargeDCUnet20()
        
        window = torch.hann_window(n_fft)
        n = torch.arange(n_fft).unsqueeze(1)
        k = torch.arange(n_fft // 2 + 1).unsqueeze(0)
        
        W_stft = torch.exp(-2j * math.pi * k * n / n_fft)
        self.register_buffer('stft_weights_r', (W_stft.real * window.unsqueeze(1)).T.unsqueeze(1))
        self.register_buffer('stft_weights_i', (W_stft.imag * window.unsqueeze(1)).T.unsqueeze(1))
        
        scale = torch.ones(n_fft // 2 + 1)
        scale[1:-1] = 2.0
        scale /= n_fft
        
        W_istft_r = (torch.cos(2 * math.pi * k * n / n_fft) * window.unsqueeze(1) * scale.unsqueeze(0) / 1.5).T.unsqueeze(1)
        W_istft_i = (-torch.sin(2 * math.pi * k * n / n_fft) * window.unsqueeze(1) * scale.unsqueeze(0) / 1.5).T.unsqueeze(1)
        self.register_buffer('istft_weights_r', W_istft_r)
        self.register_buffer('istft_weights_i', W_istft_i)

    def forward(self, x):
        original_T = x.size(-1)
        if x.dim() == 2: x = x.unsqueeze(1)
        
        # STFT
        X_r = F.conv1d(x, self.stft_weights_r, stride=self.hop_length)
        X_i = F.conv1d(x, self.stft_weights_i, stride=self.hop_length)
        X_stft = torch.complex(X_r, X_i)
        
        X_cut = X_stft[:, :-1, :].unsqueeze(1) 
        T_f = X_cut.size(-1)
        pad_t = (512 - (T_f % 512)) % 512
        if pad_t > 0: X_cut = F.pad(X_cut, (0, pad_t))
            
        O_cut = self.dcunet(X_cut)
        
        if pad_t > 0: O_cut = O_cut[..., :-pad_t]
        O = F.pad(O_cut, (0, 0, 0, 1)).squeeze(1)
        
        mag_O = torch.abs(O)
        M_hat = torch.tanh(mag_O) * (O / (mag_O + 1e-8))
        Y_hat_stft = M_hat * X_stft
        
        out_r = F.conv_transpose1d(Y_hat_stft.real, self.istft_weights_r, stride=self.hop_length)
        out_i = F.conv_transpose1d(Y_hat_stft.imag, self.istft_weights_i, stride=self.hop_length)
        y_hat = (out_r + out_i).squeeze(1)
        
        if y_hat.size(-1) > original_T: y_hat = y_hat[..., :original_T]
        elif y_hat.size(-1) < original_T: y_hat = F.pad(y_hat, (0, original_T - y_hat.size(-1)))
            
        return y_hat