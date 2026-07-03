"""
MacroAD: Main model class that composes all modules.
Multi-scale Anomaly detection with Cross-scale Reconstruction and Adaptive Decomposition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import AdaIN, RevIN
from .decomposition import LearnableWaveletDecomposition
from .embedding import PatchEmbedding, PositionalEmbedding
from .encoder import MambaEncoder
from .decoder import MambaDecoder
from .graph import TemporalGraphAttention
from .memory import Router, Extractor, ContextNet


class MacroAD(nn.Module):
    """
    MacroAD: Modular architecture for time series anomaly detection.

    Pipeline:
        Input -> Normalization -> Multi-Scale Decomposition -> Patch Embedding
        -> Mamba Encoder -> Graph Attention -> Context Memory -> Mamba Decoder
        -> Anomaly Scoring -> Output
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        d_model = configs.d_model
        seq_len = configs.seq_len
        patch_len = configs.patch_len

        # --- Stage 1: Normalization ---
        self.use_adain = getattr(configs, 'use_adain', False)
        self.use_revin = getattr(configs, 'use_revin', False)
        if self.use_adain:
            self.normalizer = AdaIN(
                num_freq_bins=getattr(configs, 'adain_freq_bins', 16),
                hidden_dim=getattr(configs, 'adain_hidden_dim', 64)
            )
        elif self.use_revin:
            self.normalizer = RevIN()
        else:
            self.normalizer = None

        # --- Stage 2: Multi-Scale Decomposition ---
        ms_kernels = configs.ms_kernels
        self.n_scales = len(ms_kernels)
        self.decomposition = LearnableWaveletDecomposition(
            kernels=ms_kernels,
            use_detail=getattr(configs, 'ms_use_detail', True)
        )
        self.ms_t_lens = self.decomposition.get_output_lengths(seq_len)

        # --- Stage 3: Patch Embedding (per-scale) ---
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(d_model, patch_len) for _ in range(self.n_scales)
        ])
        self.pos_embedding = PositionalEmbedding(d_model, learnable=getattr(configs, 'learnable_pe', False))
        if getattr(configs, 'ms_use_detail', True):
            self.detail_proj = nn.Linear(d_model, d_model)

        # Scale embeddings — distinguish tokens from different temporal resolutions
        self.scale_embeddings = nn.Parameter(torch.randn(self.n_scales, 1, d_model) * 0.02)

        # Compute total number of patches across all scales
        self._n_patches_per_scale = []
        for i, t_len in enumerate(self.ms_t_lens[:-1]):
            n_patches = (t_len + patch_len - 1) // patch_len
            self._n_patches_per_scale.append(n_patches)
        self._total_patches = sum(self._n_patches_per_scale)

        # --- Stage 4: Encoder ---
        self.encoder = MambaEncoder(
            d_model=d_model,
            n_layers=configs.e_layers,
            d_state=getattr(configs, 'd_state', 16),
            d_conv=getattr(configs, 'd_conv', 4),
            expand=getattr(configs, 'expand', 2),
            dropout=configs.ff_dropout,
            use_mamba2=getattr(configs, 'use_mamba2', False),
            n_heads=getattr(configs, 'mamba2_n_heads', 4)
        )

        # --- Stage 5: Graph Attention ---
        self.use_graph = getattr(configs, 'use_gnn', True)
        if self.use_graph:
            self.graph_attn = TemporalGraphAttention(
                d_model=d_model,
                n_hops=getattr(configs, 'tgat_n_hops', 2),
                dropout=configs.attn_dropout
            )

        # --- Stage 6: Context Memory ---
        expand = getattr(configs, 'expand', 2)
        d_state = getattr(configs, 'd_state', 16)
        d_inner = int(expand * d_model)
        d_state_input = d_inner * d_state

        use_hier = getattr(configs, 'use_hier_memory', False)
        hier_config = {
            'short_size': getattr(configs, 'hier_short_size', 16),
            'short_decay': getattr(configs, 'hier_short_decay', 0.8),
            'medium_size': getattr(configs, 'hier_medium_size', 48),
            'medium_decay': getattr(configs, 'hier_medium_decay', 0.95),
            'long_size': getattr(configs, 'hier_long_size', 96),
            'long_decay': getattr(configs, 'hier_long_decay', 0.995),
        } if use_hier else None

        extractor = Extractor(
            d_model=d_model,
            n_layers=configs.m_layers,
            n_heads=configs.n_heads,
            dropout=configs.ff_dropout,
            bank_size=configs.bank_size,
            query_len=configs.query_len,
            decay=configs.decay,
            epsilon=configs.epsilon,
            use_hierarchical=use_hier,
            hier_config=hier_config
        )

        self.context_net = ContextNet(
            router=Router(
                seq_len=self.ms_t_lens[-1],
                n_query=configs.n_query,
                topk=configs.topk,
                d_state_input=d_state_input
            ),
            querys=nn.Parameter(torch.randn(configs.n_query, configs.query_len, d_model) * 0.02),
            extractor=extractor
        )

        # Normalize context before feeding to decoder cross-attention
        self.context_norm = nn.LayerNorm(d_model)

        # --- Stage 7: Decoder ---
        # Output projection maps from d_model to patch_len per token,
        # then we reconstruct exactly _total_patches * patch_len points
        self.decoder = MambaDecoder(
            d_model=d_model,
            n_layers=configs.d_layers,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.ff_dropout,
            activation=configs.activation,
            patch_len=patch_len
        )

        # Learned upsampling to handle any mismatch between decoder output and seq_len
        dec_output_len = self._total_patches * patch_len
        if dec_output_len != seq_len:
            self.upsample = nn.Linear(dec_output_len, seq_len)
        else:
            self.upsample = None

        # --- Dual-Path Decoder ---
        self.use_dual_decoder = getattr(configs, 'use_dual_decoder', False)
        self.pred_weight = getattr(configs, 'pred_weight', 0.3)
        self.pred_horizon = getattr(configs, 'pred_horizon', 12)

    def _normalize(self, x):
        if self.normalizer is not None:
            return self.normalizer(x, 'norm')
        return x

    def _denormalize(self, x):
        if self.normalizer is not None:
            return self.normalizer(x, 'denorm')
        return x

    def _forward(self, x_enc, scoring_mode=False):
        bs, t, c = x_enc.shape

        # Normalize
        x_enc = self._normalize(x_enc)

        # Channel independence
        x_enc = x_enc.permute(0, 2, 1).reshape(bs * c, t, 1)
        router_input = x_enc

        # Multi-scale decomposition
        x_ci = x_enc.permute(0, 2, 1)  # [B*C, 1, T]
        decomp_result = self.decomposition(x_ci)
        if isinstance(decomp_result, tuple):
            approx_list, detail_list = decomp_result
        else:
            approx_list, detail_list = decomp_result, None

        # Ground truth: full input at original resolution
        ms_gt = x_enc.reshape(bs, c, -1).permute(0, 2, 1)  # [bs, t, c]

        # Per-scale patch embedding with scale tokens
        embedded_scales = []
        for i in range(self.n_scales):
            approx_i = approx_list[i].permute(0, 2, 1)  # [B*C, 1, T_i]
            emb = self.patch_embeddings[i](approx_i)

            # Fuse detail coefficients
            if detail_list is not None and hasattr(self, 'detail_proj'):
                detail_i = detail_list[i].permute(0, 2, 1)
                detail_emb = self.patch_embeddings[i](detail_i)
                emb = emb + self.detail_proj(detail_emb)

            # Add scale embedding
            emb = emb + self.scale_embeddings[i]
            embedded_scales.append(emb)

        ms_x_enc = torch.cat(embedded_scales, dim=1)

        # Positional encoding
        pos_emb = self.pos_embedding(ms_x_enc.shape[1])
        ms_x_enc = ms_x_enc + pos_emb

        # Encoder
        ms_x_enc, states = self.encoder(ms_x_enc)
        last_state = states[-1]

        # Graph attention
        if self.use_graph:
            ms_x_enc = self.graph_attn(ms_x_enc, bs, c)

        # Context memory
        distances, context = self.context_net(router_input, ms_x_enc, state=last_state, scoring_mode=scoring_mode)
        # context is [context_len, d_model] — fixed size regardless of batch
        context = self.context_norm(context)
        context = context.unsqueeze(0).expand(bs * c, -1, -1)
        distances = distances.reshape(bs, c, 1).permute(0, 2, 1)

        # Decoder
        ms_x_dec = self.decoder(ms_x_enc, context)
        # ms_x_dec: [B*C, total_patches * patch_len]  (from Flatten in decoder projection)
        ms_x_dec = ms_x_dec.unsqueeze(-1)  # [B*C, dec_len, 1]

        # Learned upsampling if decoder output != seq_len
        if self.upsample is not None:
            ms_x_dec = ms_x_dec.squeeze(-1)  # [B*C, dec_len]
            ms_x_dec = self.upsample(ms_x_dec)  # [B*C, t]
            ms_x_dec = ms_x_dec.unsqueeze(-1)  # [B*C, t, 1]
        ms_x_dec = ms_x_dec.reshape(bs, c, t).permute(0, 2, 1)  # [bs, t, c]

        # Denormalize
        ms_x_dec = self._denormalize(ms_x_dec)
        ms_gt = self._denormalize(ms_gt)

        return ms_gt, ms_x_dec, distances

    def forward(self, x_enc):
        """Training forward: returns (loss, query_distance, reconstruction, ground_truth)."""
        ms_gt, ms_x_dec, distances = self._forward(x_enc)

        loss = F.mse_loss(ms_x_dec, ms_gt)

        # Dual-path prediction emphasis
        if self.use_dual_decoder and self.training:
            h = self.pred_horizon
            if ms_gt.shape[1] > h:
                loss_pred = F.mse_loss(ms_x_dec[:, -h:, :], ms_gt[:, -h:, :])
                diff_dec = ms_x_dec[:, 1:, :] - ms_x_dec[:, :-1, :]
                diff_gt = ms_gt[:, 1:, :] - ms_gt[:, :-1, :]
                loss_temporal = F.mse_loss(diff_dec[:, -h:, :], diff_gt[:, -h:, :])
                loss = loss + self.pred_weight * (loss_pred + loss_temporal)

        return loss, torch.mean(distances), ms_x_dec, ms_gt

    def infer(self, x_enc):
        """Inference: returns (anomaly_scores [B, T, C], query_distances [B, 1, C])."""
        ms_gt, ms_x_dec, distances = self._forward(x_enc, scoring_mode=False)
        scores = F.mse_loss(ms_x_dec, ms_gt, reduction="none")
        return scores, distances
