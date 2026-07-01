"""
Multi-scale decomposition modules for MacroAD.
- LearnableWaveletDecomposition: Adaptive filters with detail coefficients
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableWaveletDecomposition(nn.Module):
    """Learnable multi-scale decomposition with low-pass and high-pass filters.
    Replaces fixed average pooling with data-adaptive filter banks."""
    def __init__(self, kernels, use_detail=True):
        super().__init__()
        self.kernels = kernels
        self.use_detail = use_detail
        self.n_scales = len(kernels)

        self.low_pass_filters = nn.ParameterList()
        for k in kernels:
            w = torch.ones(1, 1, k) / k
            self.low_pass_filters.append(nn.Parameter(w))

        if use_detail:
            self.high_pass_filters = nn.ParameterList()
            for k in kernels:
                w = torch.zeros(1, 1, k)
                half = k // 2
                w[0, 0, :half] = 1.0 / half
                w[0, 0, half:] = -1.0 / half
                self.high_pass_filters.append(nn.Parameter(w))

    def forward(self, x):
        """
        x: [B, C, T] (channels-first)
        Returns: (approximations, details) or approximations only
        """
        approx_list = []
        detail_list = []
        C = x.shape[1]

        for idx, kernel in enumerate(self.kernels):
            pad_size = kernel - 1
            padded = F.pad(x, (0, pad_size), mode="replicate")

            filt = self.low_pass_filters[idx]
            filt_norm = filt / (filt.sum() + 1e-8)
            filt_expanded = filt_norm.expand(C, 1, -1)
            approx = F.conv1d(padded, filt_expanded, stride=kernel, groups=C)
            approx_list.append(approx.permute(0, 2, 1))

            if self.use_detail:
                filt_h = self.high_pass_filters[idx]
                filt_h_expanded = filt_h.expand(C, 1, -1)
                detail = F.conv1d(padded, filt_h_expanded, stride=kernel, groups=C)
                detail_list.append(detail.permute(0, 2, 1))

        if self.use_detail:
            return approx_list, detail_list
        return approx_list

    @torch.no_grad()
    def get_output_lengths(self, input_len):
        """Compute output lengths for each scale."""
        dummy = torch.ones(1, 1, input_len)
        results = self.forward(dummy)
        if isinstance(results, tuple):
            approx_list = results[0]
        else:
            approx_list = results
        return [a.shape[1] for a in approx_list] + [input_len]
