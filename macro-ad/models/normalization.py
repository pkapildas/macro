"""
Normalization modules for MacroAD.
- RevIN: Standard reversible instance normalization
- AdaIN: Frequency-conditioned adaptive instance normalization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022)."""
    def __init__(self, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise NotImplementedError

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        return (x - self.mean) / self.stdev

    def _denormalize(self, x):
        return x * self.stdev + self.mean


class AdaIN(nn.Module):
    """Adaptive Instance Normalization conditioned on frequency content.
    Preserves anomaly-relevant distribution information via learned affine params."""
    def __init__(self, num_freq_bins=16, hidden_dim=64, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_freq_bins = num_freq_bins
        self.freq_mlp = nn.Sequential(
            nn.Linear(num_freq_bins, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )
        nn.init.zeros_(self.freq_mlp[-1].weight)
        nn.init.zeros_(self.freq_mlp[-1].bias)

    def forward(self, x, mode: str):
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise NotImplementedError

    def _compute_freq_signature(self, x):
        x_mean_c = x.mean(dim=-1)
        x_fft = torch.fft.rfft(x_mean_c, dim=-1)
        magnitudes = torch.abs(x_fft).unsqueeze(1)
        binned = F.interpolate(magnitudes, size=self.num_freq_bins, mode='linear', align_corners=False)
        return torch.log1p(binned.squeeze(1))

    def _normalize(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

        freq_sig = self._compute_freq_signature(x)
        params = self.freq_mlp(freq_sig)
        self.ada_gamma = params[:, 0:1].unsqueeze(1) + 1.0
        self.ada_beta = params[:, 1:2].unsqueeze(1)

        x = (x - self.mean) / self.stdev
        return x * self.ada_gamma + self.ada_beta

    def _denormalize(self, x):
        x = (x - self.ada_beta) / (self.ada_gamma + self.eps)
        return x * self.stdev + self.mean
