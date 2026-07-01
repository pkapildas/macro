"""
Embedding modules for MacroAD.
- PatchEmbedding: Tokenizes time series into patches
- PositionalEmbedding: Sinusoidal or learnable position encoding
"""
import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Projects time series patches to d_model embedding space."""
    def __init__(self, d_model, patch_len, stride=None, dropout=0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride or patch_len
        self.padding = nn.ReplicationPad1d((0, patch_len - 1))
        self.projection = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, C, T] -> patches: [B*C, n_patches, d_model]"""
        x = self.padding(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return self.dropout(self.projection(x))

    @torch.no_grad()
    def get_num_patches(self, input_lens):
        return [(l + self.patch_len - 1) // self.stride for l in input_lens]


class PositionalEmbedding(nn.Module):
    """Sinusoidal or learnable positional encoding."""
    def __init__(self, d_model, max_len=5000, learnable=False):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.uniform_(self.pe, -0.02, 0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).float().unsqueeze(1)
            div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, seq_len):
        return self.pe[:, :seq_len]
