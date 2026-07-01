"""
Memory and routing modules for MacroAD.
- MemoryTier: Single EMA memory bank
- HierarchicalMemory: 3-tier (short/medium/long) with learned fusion
- Router: FFT + state-based query selection
- ContextNet: Composes Router + Query Library + Memory
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Routes input to relevant query prototypes via FFT + Mamba state."""
    def __init__(self, seq_len, n_query, topk=10, d_state_input=None):
        super().__init__()
        self.k = topk
        self.fc = nn.Sequential(nn.Flatten(-2), nn.Linear(seq_len, n_query))

        self.d_state_input = d_state_input
        if d_state_input is not None:
            self.state_proj = nn.Linear(d_state_input, n_query)

    def forward(self, x, state=None):
        bs, t, c = x.shape
        x_freq = torch.fft.rfft(x, dim=1, n=t)
        _, indices = torch.topk(x_freq.abs(), self.k, dim=1)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing="ij")
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        mask = torch.zeros_like(x_freq, dtype=torch.bool)
        mask[index_tuple] = True
        x_freq[~mask] = torch.tensor(0.0+0j, device=x_freq.device)
        x = torch.fft.irfft(x_freq, dim=1, n=t)

        logits = self.fc(x)

        if state is not None and hasattr(self, 'state_proj'):
            state_flat = state.reshape(bs, -1)
            logits = logits + self.state_proj(state_flat)

        return F.gumbel_softmax(logits, tau=1, hard=True)


class MemoryTier(nn.Module):
    """Single EMA memory bank with soft attention routing."""
    def __init__(self, tier_size, query_len, d_model, decay, epsilon=1e-5):
        super().__init__()
        self.tier_size = tier_size
        self.query_len = query_len
        self.d_model = d_model
        self.decay = decay
        self.epsilon = epsilon

        self.register_buffer("memory", torch.randn(tier_size, query_len, d_model))
        self.register_buffer("ema_count", torch.ones(tier_size))
        self.register_buffer("ema_dw", torch.zeros(tier_size, query_len, d_model))

    def update(self, q):
        _, q_len, d = q.shape
        q_flat = q.reshape(-1, q_len * d)
        g_flat = self.memory.reshape(-1, q_len * d)

        scores = torch.matmul(q_flat, g_flat.t()) / (g_flat.shape[-1] ** 0.5)
        encodings = F.softmax(scores, dim=-1)

        q_context = torch.einsum("bn,nqd->bqd", encodings, self.memory)
        q_hat = torch.einsum("bn,bqd->nqd", encodings, q)
        query_latent_distances = torch.mean(F.mse_loss(q_context.detach(), q, reduction="none"), dim=(1, 2))

        if self.training:
            with torch.no_grad():
                N = self.tier_size
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
                n = torch.sum(self.ema_count)
                self.ema_count = (self.ema_count + self.epsilon) / (n + N * self.epsilon) * n
                dw = torch.einsum("bn,bqd->nqd", encodings, q)
                self.ema_dw = self.decay * self.ema_dw + (1 - self.decay) * dw
                self.memory = self.ema_dw / self.ema_count.unsqueeze(-1).unsqueeze(-1)

        return query_latent_distances, q_hat


class HierarchicalMemory(nn.Module):
    """3-tier memory: short-term (fast adapt), medium-term, long-term (stable)."""
    def __init__(self, query_len, d_model,
                 short_size=16, short_decay=0.8,
                 medium_size=48, medium_decay=0.95,
                 long_size=96, long_decay=0.995, epsilon=1e-5):
        super().__init__()
        self.query_len = query_len
        self.d_model = d_model

        self.short_term = MemoryTier(short_size, query_len, d_model, short_decay, epsilon)
        self.medium_term = MemoryTier(medium_size, query_len, d_model, medium_decay, epsilon)
        self.long_term = MemoryTier(long_size, query_len, d_model, long_decay, epsilon)

        self.tier_gate = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 3), nn.Softmax(dim=-1)
        )
        self.tier_proj_short = nn.Linear(d_model, d_model)
        self.tier_proj_medium = nn.Linear(d_model, d_model)
        self.tier_proj_long = nn.Linear(d_model, d_model)

    def update_context(self, q):
        dist_s, q_hat_s = self.short_term.update(q)
        dist_m, q_hat_m = self.medium_term.update(q)
        dist_l, q_hat_l = self.long_term.update(q)

        query_latent_distances = 0.5 * dist_s + 0.3 * dist_m + 0.2 * dist_l

        ctx_s = self.tier_proj_short(q_hat_s + self.short_term.memory.detach() - q_hat_s.detach())
        ctx_m = self.tier_proj_medium(q_hat_m + self.medium_term.memory.detach() - q_hat_m.detach())
        ctx_l = self.tier_proj_long(q_hat_l + self.long_term.memory.detach() - q_hat_l.detach())

        q_pooled = q.mean(dim=1)
        tier_weights = self.tier_gate(q_pooled)
        w_s, w_m, w_l = tier_weights[:, 0].mean(), tier_weights[:, 1].mean(), tier_weights[:, 2].mean()

        context = torch.cat([
            (ctx_s * w_s).view(-1, self.d_model),
            (ctx_m * w_m).view(-1, self.d_model),
            (ctx_l * w_l).view(-1, self.d_model),
        ], dim=0)
        return query_latent_distances, context

    def concat_context(self):
        ctx_s = self.tier_proj_short(self.short_term.memory)
        ctx_m = self.tier_proj_medium(self.medium_term.memory)
        ctx_l = self.tier_proj_long(self.long_term.memory)
        return torch.cat([ctx_s.view(-1, self.d_model), ctx_m.view(-1, self.d_model), ctx_l.view(-1, self.d_model)], dim=0)


class Extractor(nn.Module):
    """Cross-attention refinement + memory bank update."""
    def __init__(self, d_model, n_layers=2, n_heads=4, d_ff=None, dropout=0.1,
                 bank_size=32, query_len=5, decay=0.95, epsilon=1e-5,
                 use_hierarchical=False, hier_config=None):
        super().__init__()
        self.d_model = d_model
        self.query_len = query_len
        self.use_hierarchical = use_hierarchical

        d_ff = d_ff or 4 * d_model
        self.layers = nn.ModuleList([
            ExtractorLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        if use_hierarchical and hier_config:
            self.memory = HierarchicalMemory(query_len, d_model, **hier_config)
        else:
            self.memory = MemoryTier(bank_size, query_len, d_model, decay, epsilon)

    def forward(self, q, local_repr):
        for layer in self.layers:
            q = layer(q, local_repr)

        if self.use_hierarchical:
            distances, context = self.memory.update_context(q)
        else:
            distances, q_hat = self.memory.update(q)
            context = (q_hat + self.memory.memory.detach() - q_hat.detach()).view(-1, self.d_model)

        return distances, context

    def get_context_for_inference(self):
        if self.use_hierarchical:
            return self.memory.concat_context()
        return self.memory.memory.view(-1, self.d_model)


class ExtractorLayer(nn.Module):
    """Cross-attention + FFN for query refinement."""
    def __init__(self, d_model, n_heads=4, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, local_repr):
        attn_out, _ = self.cross_attn(q, local_repr, local_repr)
        q = self.norm1(q + self.dropout(attn_out))
        q = self.norm2(q + self.ffn(q))
        return q


class ContextNet(nn.Module):
    """Composes Router + Query Library + Extractor."""
    def __init__(self, router, querys, extractor):
        super().__init__()
        self.router = router
        self.querys = querys
        self.extractor = extractor

    def forward(self, x_enc, local_repr, state=None):
        q_indices = self.router(x_enc, state=state)
        q = torch.einsum('bn,nqd->bqd', q_indices, self.querys)
        distances, context = self.extractor(q, local_repr)
        return distances, context
