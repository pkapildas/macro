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
        device = x.device
        x_freq = torch.fft.rfft(x, dim=1, n=t)
        _, indices = torch.topk(x_freq.abs(), self.k, dim=1)
        mesh_a, mesh_b = torch.meshgrid(
            torch.arange(x_freq.size(0), device=device),
            torch.arange(x_freq.size(2), device=device),
            indexing="ij"
        )
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        mask = torch.zeros_like(x_freq, dtype=torch.bool)
        mask[index_tuple] = True
        x_freq = x_freq.masked_fill(~mask, 0.0)
        x = torch.fft.irfft(x_freq, dim=1, n=t)

        logits = self.fc(x)

        if state is not None and hasattr(self, 'state_proj'):
            state_flat = state.reshape(bs, -1)
            logits = logits + self.state_proj(state_flat)

        if self.training:
            return F.gumbel_softmax(logits, tau=1, hard=True)
        else:
            idx = logits.argmax(dim=-1)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(-1, idx.unsqueeze(-1), 1.0)
            return one_hot


class MemoryTier(nn.Module):
    """Single EMA memory bank with soft attention routing."""
    def __init__(self, tier_size, query_len, d_model, decay, epsilon=1e-5):
        super().__init__()
        self.tier_size = tier_size
        self.query_len = query_len
        self.d_model = d_model
        self.decay = decay
        self.epsilon = epsilon

        self.register_buffer("memory", torch.randn(tier_size, query_len, d_model) * 0.02)
        self.register_buffer("ema_count", torch.ones(tier_size))
        self.register_buffer("ema_dw", torch.zeros(tier_size, query_len, d_model))

    def update(self, q, use_straight_through=True):
        """
        q: [B, query_len, d_model]
        Returns: (distances [B], context [tier_size, query_len, d_model])
        Context shape is ALWAYS [tier_size, query_len, d_model] regardless of batch size.

        use_straight_through: if True, use memory directly (for training/validation loss).
                              if False, use input-dependent weighted readout (for anomaly scoring).
        """
        _, q_len, d = q.shape
        q_flat = q.reshape(-1, q_len * d)
        g_flat = self.memory.reshape(-1, q_len * d)

        # Clamp pre-softmax scores to prevent overflow
        scores = torch.matmul(q_flat, g_flat.t()) / (g_flat.shape[-1] ** 0.5)
        scores = torch.clamp(scores, min=-30.0, max=30.0)
        encodings = F.softmax(scores, dim=-1)

        # q_context: what the memory returns for each input query
        q_context = torch.einsum("bn,nqd->bqd", encodings, self.memory)
        query_latent_distances = torch.mean(
            F.mse_loss(q_context.detach(), q, reduction="none"), dim=(1, 2)
        )

        if self.training:
            with torch.no_grad():
                N = self.tier_size
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
                n = torch.sum(self.ema_count)
                self.ema_count = (self.ema_count + self.epsilon) / (n + N * self.epsilon) * n
                self.ema_count = torch.clamp(self.ema_count, min=self.epsilon)
                dw = torch.einsum("bn,bqd->nqd", encodings, q)
                self.ema_dw = self.decay * self.ema_dw + (1 - self.decay) * dw
                self.memory = self.ema_dw / self.ema_count.unsqueeze(-1).unsqueeze(-1)
                self.memory = torch.clamp(self.memory, min=-5.0, max=5.0)

        if use_straight_through:
            # Straight-through: context = memory (with gradient path during training)
            q_hat = torch.einsum("bn,bqd->nqd", encodings, q)
            context = q_hat + self.memory.detach() - q_hat.detach()
        else:
            # Input-dependent: per-query weighted readout reshaped to bank shape
            # Use activation pattern to produce per-slot context
            activation = encodings.mean(dim=0)  # [N] avg activation per slot
            activation = (activation / (activation.max() + 1e-8)).unsqueeze(-1).unsqueeze(-1)
            # Blend: highly activated slots get more input influence
            q_mean = q_context.mean(dim=0, keepdim=True).expand_as(self.memory)
            context = (1 - activation) * self.memory + activation * q_mean

        return query_latent_distances, context


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
        dist_s, ctx_s_raw = self.short_term.update(q)
        dist_m, ctx_m_raw = self.medium_term.update(q)
        dist_l, ctx_l_raw = self.long_term.update(q)

        query_latent_distances = 0.5 * dist_s + 0.3 * dist_m + 0.2 * dist_l

        ctx_s = self.tier_proj_short(ctx_s_raw)
        ctx_m = self.tier_proj_medium(ctx_m_raw)
        ctx_l = self.tier_proj_long(ctx_l_raw)

        # Per-sample gate weights
        q_pooled = q.mean(dim=1)
        tier_weights = self.tier_gate(q_pooled)  # [B, 3]
        w_s = tier_weights[:, 0].mean()
        w_m = tier_weights[:, 1].mean()
        w_l = tier_weights[:, 2].mean()

        context = torch.cat([
            ctx_s.view(-1, self.d_model) * w_s,
            ctx_m.view(-1, self.d_model) * w_m,
            ctx_l.view(-1, self.d_model) * w_l,
        ], dim=0)
        return query_latent_distances, context

    def get_context(self):
        ctx_s = self.tier_proj_short(self.short_term.memory)
        ctx_m = self.tier_proj_medium(self.medium_term.memory)
        ctx_l = self.tier_proj_long(self.long_term.memory)
        return torch.cat([
            ctx_s.view(-1, self.d_model),
            ctx_m.view(-1, self.d_model),
            ctx_l.view(-1, self.d_model)
        ], dim=0)


class Extractor(nn.Module):
    """Cross-attention refinement + memory bank update."""
    def __init__(self, d_model, n_layers=2, n_heads=4, d_ff=None, dropout=0.1,
                 bank_size=32, query_len=5, decay=0.95, epsilon=1e-5,
                 use_hierarchical=False, hier_config=None):
        super().__init__()
        self.d_model = d_model
        self.query_len = query_len
        self.bank_size = bank_size
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

    def forward(self, q, local_repr, scoring_mode=False):
        for layer in self.layers:
            q = layer(q, local_repr)

        if self.use_hierarchical:
            distances, context = self.memory.update_context(q)
        else:
            distances, context = self.memory.update(q, use_straight_through=not scoring_mode)
            # context is [bank_size, query_len, d_model] — flatten to [bank_size*query_len, d_model]
            context = context.view(-1, self.d_model)

        return distances, context

    def get_context_size(self):
        """Returns the fixed context sequence length."""
        if self.use_hierarchical:
            total = (self.memory.short_term.tier_size +
                     self.memory.medium_term.tier_size +
                     self.memory.long_term.tier_size)
            return total * self.query_len
        return self.bank_size * self.query_len


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

    def forward(self, x_enc, local_repr, state=None, scoring_mode=False):
        q_indices = self.router(x_enc, state=state)
        q = torch.einsum('bn,nqd->bqd', q_indices, self.querys)
        distances, context = self.extractor(q, local_repr, scoring_mode=scoring_mode)
        return distances, context
