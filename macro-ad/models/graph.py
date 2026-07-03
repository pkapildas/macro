"""
Graph modules for MacroAD.
- TemporalGraphAttention: Multi-hop message passing with learned adjacency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGraphAttention(nn.Module):
    """Multi-hop graph attention with attention-pooled node features.
    Learns inter-variable relationships for multivariate anomaly detection."""
    def __init__(self, d_model, n_hops=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_hops = n_hops

        # Attention-based pooling over time dimension (scale-aware)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.Linear(d_model, 1)

        self.edge_q = nn.Linear(d_model, d_model)
        self.edge_k = nn.Linear(d_model, d_model)

        self.msg_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_hops)
        ])
        self.hop_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_hops)
        ])

        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, bs, c):
        """
        x: [B*C, T, D] -- channel-independent encoded features
        bs: batch size
        c: number of channels
        Returns: [B*C, T, D]
        """
        _, T, D = x.shape
        x_4d = x.view(bs, c, T, D)

        # Attention-weighted pooling over time (respects scale structure)
        attn_scores = self.pool_attn(x_4d).squeeze(-1)  # [BS, C, T]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [BS, C, T]
        nodes = torch.einsum("bct,bctd->bcd", attn_weights, x_4d)  # [BS, C, D]

        # Compute adjacency
        Q = self.edge_q(nodes)
        K = self.edge_k(nodes)
        A = torch.bmm(Q, K.transpose(-1, -2)) / (D ** 0.5)
        A = F.softmax(A, dim=-1)  # [BS, C, C]

        # Multi-hop message passing
        h = nodes
        for hop in range(self.n_hops):
            msg = torch.bmm(A, h)
            msg = self.msg_layers[hop](msg)
            msg = self.dropout(F.gelu(msg))
            h = self.hop_norms[hop](h + msg)

        # Broadcast back to time + residual
        out = h.unsqueeze(2).expand(-1, -1, T, -1)  # [BS, C, T, D]
        result = self.output_norm(x_4d + out)
        return result.reshape(bs * c, T, D)
