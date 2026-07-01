"""
Decoder modules for MacroAD.
- MambaDecoder: Mamba self-mixing + Cross-Attention + FFN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MambaBlock


class DecoderLayer(nn.Module):
    """Mamba self-mixing + Cross-Attention to context + FFN."""
    def __init__(self, d_model, n_heads=4, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.self_mixing = MambaBlock(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, context, x_mask=None):
        # Self-mixing (Mamba)
        mamba_out, _ = self.self_mixing(x)
        x = self.norm1(x + mamba_out)

        # Cross-attention to context
        attn_out, _ = self.cross_attn(x, context, context)
        x = self.norm2(x + self.dropout(attn_out))

        # FFN
        x = self.norm3(x + self.ffn(x))
        return x


class MambaDecoder(nn.Module):
    """Multi-layer Mamba decoder with cross-attention to memory context."""
    def __init__(self, d_model, n_layers=2, n_heads=4, d_ff=None,
                 dropout=0.1, activation="gelu", patch_len=6):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, patch_len),
            nn.Flatten(-2)
        )

    def forward(self, x, context, mask=None):
        for layer in self.layers:
            x = layer(x, context, mask)
        x = self.norm(x)
        return self.projection(x)
