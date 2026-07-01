"""
Encoder modules for MacroAD.
- MambaEncoder: Original Mamba-1 sequential scan (stable)
- Mamba2Encoder: Multi-head Mamba with per-head B/C/A (requires CUDA)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MambaBlock(nn.Module):
    """Mamba-1: Selective State Space Model with sequential scan."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + 1, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        nn.init.uniform_(self.dt_proj.bias, 0.01, 0.1)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def ssm(self, x):
        B, L, D = x.shape
        x_proj = self.x_proj(x)
        delta = F.softplus(self.dt_proj(x_proj[:, :, 0:1]))
        delta = torch.clamp(delta, min=1e-4, max=0.1)
        B_ssm = x_proj[:, :, 1:1+self.d_state]
        C_ssm = x_proj[:, :, 1+self.d_state:]

        A = -torch.exp(self.A_log)
        A = torch.clamp(A, min=-10.0, max=-0.001)
        dA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state))
        dB = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)

        x_us = x.unsqueeze(-1)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_us[:, t]
            h = torch.clamp(h, min=-10.0, max=10.0)
            y_t = torch.sum(h * C_ssm[:, t].unsqueeze(1), dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return y + x * self.D.view(1, 1, -1), h

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_inner = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_inner = self.act(x_inner)
        y, h_last = self.ssm(x_inner)
        y = y * self.act(z)
        return self.dropout(self.out_proj(y)), h_last


class Mamba2Block(nn.Module):
    """Multi-head Mamba with per-head B/C/A projections.
    Note: Requires CUDA for stable training beyond 25 epochs."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads
        self.d_inner = int(expand * d_model)
        self.head_dim = self.d_inner // n_heads

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, n_heads * (2 * d_state + 1), bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)
        self.dt_proj = nn.Linear(n_heads, self.d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        nn.init.uniform_(self.dt_proj.bias, 0.01, 0.1)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def ssm_multihead(self, x):
        B, L, D = x.shape
        proj = self.x_proj(x).view(B, L, self.n_heads, 2 * self.d_state + 1)
        dt_raw = proj[:, :, :, 0]
        delta = F.softplus(self.dt_proj(dt_raw))
        delta = torch.clamp(delta, min=1e-4, max=0.1)
        B_ssm = proj[:, :, :, 1:1+self.d_state]
        C_ssm = proj[:, :, :, 1+self.d_state:]

        B_full = B_ssm.unsqueeze(3).expand(-1, -1, -1, self.head_dim, -1).reshape(B, L, self.d_inner, self.d_state)
        C_full = C_ssm.unsqueeze(3).expand(-1, -1, -1, self.head_dim, -1).reshape(B, L, self.d_inner, self.d_state)

        A = -torch.exp(self.A_log)
        A = torch.clamp(A, min=-10.0, max=-0.001)
        dA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state))
        dB = delta.unsqueeze(-1) * B_full

        x_us = x.unsqueeze(-1)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_us[:, t]
            h = torch.clamp(h, min=-10.0, max=10.0)
            y_t = torch.sum(h * C_full[:, t], dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return y + x * self.D.view(1, 1, -1), h

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_inner = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_inner = self.act(x_inner)
        y, h_last = self.ssm_multihead(x_inner)
        y = y * self.act(z)
        return self.dropout(self.out_proj(y)), h_last


class EncoderLayer(nn.Module):
    """Single encoder layer: Mamba + LayerNorm + Residual."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1,
                 use_mamba2=False, n_heads=4):
        super().__init__()
        if use_mamba2:
            self.mamba = Mamba2Block(d_model, d_state, d_conv, expand, n_heads, dropout)
        else:
            self.mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        mamba_out, h_last = self.mamba(x)
        return self.norm(x + mamba_out), h_last


class MambaEncoder(nn.Module):
    """Multi-layer Mamba encoder. Returns encoded features + final hidden state."""
    def __init__(self, d_model, n_layers=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.1, use_mamba2=False, n_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_state, d_conv, expand, dropout, use_mamba2, n_heads)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        states = []
        for layer in self.layers:
            x, h = layer(x, attn_mask)
            states.append(h)
        return self.norm(x), states


# Alias for clarity
Mamba2Encoder = MambaEncoder
