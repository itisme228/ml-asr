import torch
import torch.nn as nn
from typing import Tuple, List

def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return torch.stack([real, imag], dim=-1)

def apply_complex_mask(X: torch.Tensor, M: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return complex_mul(X, M) + R

class BandSplitModule(nn.Module):
    def __init__(self, band_bins: List[int], feature_dim: int):
        super().__init__()
        self.band_bins = band_bins
        self.num_bands = len(band_bins)
        
        self.norms = nn.ModuleList([nn.LayerNorm(b * 2) for b in band_bins])
        self.projs = nn.ModuleList([nn.Linear(b * 2, feature_dim) for b in band_bins])
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = X.shape
        
        X_splits = torch.split(X, self.band_bins, dim=1)
        
        Z_bands = []
        for i in range(self.num_bands):
            x_i = X_splits[i]
            x_i = x_i.permute(0, 2, 1, 3).contiguous()
            x_i = x_i.view(B, T, -1)
            
            z_i = self.norms[i](x_i)
            z_i = self.projs[i](z_i)
            
            Z_bands.append(z_i)
            
        Z = torch.stack(Z_bands, dim=1)
        return Z

class TemporalModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.proj = nn.Linear(hidden_size, feature_dim)
        
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        B, num_bands, T, D = Z.shape
        
        Z_reshaped = Z.view(B * num_bands, T, D)
        
        out, _ = self.lstm(Z_reshaped)
        out = self.proj(out)
        out = out + Z_reshaped
        out = out.view(B, num_bands, T, D)
        return out

class BandModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, num_layers: int, split_index: int = 26):
        super().__init__()
        self.split_index = split_index
        
        self.low_rnn = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.low_proj = nn.Linear(hidden_size * 2, feature_dim)
        
        self.high_rnn = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.high_proj = nn.Linear(hidden_size, feature_dim)
        
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        B, num_bands, T, D = Z.shape
        Z_perm = Z.permute(0, 2, 1, 3).contiguous()
        Z_reshaped = Z_perm.view(B * T, num_bands, D)
        
        Z_low = Z_reshaped[:, :self.split_index, :]
        Z_high = Z_reshaped[:, self.split_index:, :]
        
        low_out, _ = self.low_rnn(Z_low)
        low_out = self.low_proj(low_out)
        low_out = low_out + Z_low
        
        if Z_high.size(1) > 0:
            high_out, _ = self.high_rnn(Z_high)
            high_out = self.high_proj(high_out)
            high_out = high_out + Z_high
            out_reshaped = torch.cat([low_out, high_out], dim=1)
        else:
            out_reshaped = low_out
        
        out = out_reshaped.view(B, T, num_bands, D)
        out = out.permute(0, 2, 1, 3).contiguous()
        
        return out

class MaskEstimator(nn.Module):
    def __init__(self, band_bins: List[int], feature_dim: int):
        super().__init__()
        self.num_bands = len(band_bins)
        
        self.mlps = nn.ModuleList()
        for b in band_bins:
            self.mlps.append(nn.Sequential(
                nn.Linear(feature_dim, 384),
                nn.Tanh(),
                nn.Linear(384, b * 8),
                nn.GLU(dim=-1)
            ))
            
    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, num_bands, T, D = Z.shape
        
        out_bands = []
        for i in range(self.num_bands):
            z_i = Z[:, i, :, :]
            out_i = self.mlps[i](z_i)
            
            b_i = out_i.shape[-1] // 4
            out_i = out_i.view(B, T, b_i, 4)
            
            out_i = out_i.permute(0, 2, 1, 3).contiguous()
            out_bands.append(out_i)
            
        out = torch.cat(out_bands, dim=1)
        
        M = out[..., :2]
        R = out[..., 2:]
        
        return M, R

class BSRNN(nn.Module):
    def __init__(self, freq_bins: int, feature_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        
        target_bandwidths = [200] * 20 + [500] * 6 + [2000] * 7
        total_bandwidth = sum(target_bandwidths)
        
        self.band_bins = []
        remaining_bins = freq_bins
        for bw in target_bandwidths[:-1]:
            b = max(1, int(freq_bins * (bw / total_bandwidth)))
            self.band_bins.append(b)
            remaining_bins -= b
        self.band_bins.append(remaining_bins)
        
        self.band_split = BandSplitModule(self.band_bins, feature_dim)
        self.temporal_model = TemporalModel(feature_dim, hidden_size, num_layers)
        
        self.band_model = BandModel(feature_dim, hidden_size, num_layers, split_index=26)
        
        self.mask_estimator = MaskEstimator(self.band_bins, feature_dim)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:

        Z = self.band_split(X)
        Z = self.temporal_model(Z)
        Z = self.band_model(Z)
        M, R = self.mask_estimator(Z)
        S_hat = apply_complex_mask(X, M, R)
        S_hat = 0.8 * S_hat + 0.2 * X
        return S_hat