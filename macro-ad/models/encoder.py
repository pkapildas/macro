"""
Encoder modules for MacroAD.
- MambaBlock: Selective State Space Model with optimized parallel scan
- Mamba2Block: Multi-head variant
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def parallel_scan(dA, dB_x):
    """
    Parallel associative scan for SSM recurrence.
    dA: [B, L, D, N] — discretized state matrix
    dB_x: [B, L, D, N] — input contribution (dB * x)
    Returns: hidden states [B, L, D, N]
    """
    B, L, D, N = dA.shape

    if L <= 32:
        h = torch.zeros(B, D, N, device=dA.device, dtype=dA.dtype)
        hs = []
        for t in range(L):
            h = dA[:, t] * h + dB_x[:, t]
            h = torch.clamp(h, min=-10.0, max=10.0)
            hs.append(h)
        return torch.stack(hs, dim=1)

    # Parallel scan via divide-and-conquer
    # Pad to power of 2
    L_pad = 1 << (L - 1).bit_length()
    if L_pad > L:
        pad_a = torch.ones(B, L_pad - L, D, N, device=dA.device, dtype=dA.dtype)
        pad_b = torch.zeros(B, L_pad - L, D, N, device=dA.device, dtype=dA.dtype)
        dA = torch.cat([dA, pad_a], dim=1)
        dB_x = torch.cat([dB_x, pad_b], dim=1)

    # Blelloch-style parallel prefix scan
    # Up-sweep
    a_vals = dA.clone()
    b_vals = dB_x.clone()

    steps = int(math.log2(L_pad))
    for d in range(steps):
        stride = 1 << (d + 1)
        idx = torch.arange(stride - 1, L_pad, stride, device=dA.device)
        prev_idx = idx - (stride // 2)

        b_vals[:, idx] = a_vals[:, idx] * b_vals[:, prev_idx] + b_vals[:, idx]
        a_vals[:, idx] = a_vals[:, idx] * a_vals[:, prev_idx]

    # Down-sweep
    for d in range(steps - 2, -1, -1):
        stride = 1 << (d + 1)
        idx = torch.arange(stride + (stride // 2) - 1, L_pad, stride, device=dA.device)
        prev_idx = idx - (stride // 2)

        if len(idx) > 0:
            b_vals[:, idx] = a_vals[:, idx] * b_vals[:, prev_idx] + b_vals[:, idx]
            a_vals[:, idx] = a_vals[:, idx] * a_vals[:, prev_idx]

    return b_vals[:, :L]


class MambaBlock(nn.Module):
    """Mamba-1: Selective State Space Model with parallel scan."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.pre_ssm_norm = nn.LayerNorm(self.d_inner)
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
        # Normalize before projection to prevent unbounded growth
        x_normed = self.pre_ssm_norm(x)
        x_proj = self.x_proj(x_normed)
        delta = F.softplus(self.dt_proj(x_proj[:, :, 0:1]))
        delta = torch.clamp(delta, min=1e-4, max=0.05)
        B_ssm = torch.tanh(x_proj[:, :, 1:1+self.d_state])
        C_ssm = torch.tanh(x_proj[:, :, 1+self.d_state:])

        A = -torch.exp(self.A_log)
        A = torch.clamp(A, min=-5.0, max=-0.01)

        # Compute discretized matrices
        dA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state))
        dB = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)  # [B, L, D, N]
        dB_x = dB * x.unsqueeze(-1)  # [B, L, D, N]

        # Parallel scan
        h_all = parallel_scan(dA, dB_x)

        # Output: y = C * h + D * x
        y = torch.sum(h_all * C_ssm.unsqueeze(2), dim=-1)
        h_last = h_all[:, -1]

        return y + x * self.D.view(1, 1, -1), h_last

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
    """Multi-head Mamba with per-head B/C/A projections and parallel scan."""
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
        self.pre_ssm_norm = nn.LayerNorm(self.d_inner)
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
        x_normed = self.pre_ssm_norm(x)
        proj = self.x_proj(x_normed).view(B, L, self.n_heads, 2 * self.d_state + 1)
        dt_raw = proj[:, :, :, 0]
        delta = F.softplus(self.dt_proj(dt_raw))
        delta = torch.clamp(delta, min=1e-4, max=0.05)
        B_ssm = torch.tanh(proj[:, :, :, 1:1+self.d_state])
        C_ssm = torch.tanh(proj[:, :, :, 1+self.d_state:])

        # Expand per-head to per-dim efficiently
        B_full = B_ssm.repeat_interleave(self.head_dim, dim=2)
        C_full = C_ssm.repeat_interleave(self.head_dim, dim=2)

        A = -torch.exp(self.A_log)
        A = torch.clamp(A, min=-5.0, max=-0.01)
        dA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state))
        dB_x = (delta.unsqueeze(-1) * B_full) * x.unsqueeze(-1)

        # Parallel scan
        h_all = parallel_scan(dA, dB_x)

        y = torch.sum(h_all * C_full, dim=-1)
        h_last = h_all[:, -1]

        return y + x * self.D.view(1, 1, -1), h_last

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


Mamba2Encoder = MambaEncoder
