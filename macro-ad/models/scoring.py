"""
Anomaly scoring modules for MacroAD.
- ScaleAttentionFusion: Learned multi-scale score fusion
- DistributionAwareScoring: Mahalanobis + spectral divergence scoring
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAttentionFusion(nn.Module):
    """Learned weighted fusion of per-scale anomaly scores."""
    def __init__(self, n_scales):
        super().__init__()
        self.n_scales = n_scales
        self.attn_mlp = nn.Sequential(
            nn.Linear(n_scales, 64),
            nn.Tanh(),
            nn.Linear(64, n_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, ms_score_list):
        T_final = ms_score_list[-1].shape[1]
        B, _, C = ms_score_list[-1].shape

        # Upsample all scores to finest resolution
        upsampled = []
        for score in ms_score_list:
            if score.shape[1] != T_final:
                s = score.permute(0, 2, 1)
                s = F.interpolate(s, size=T_final, mode='linear')
                upsampled.append(s.permute(0, 2, 1))
            else:
                upsampled.append(score)

        # Compute per-scale context
        scale_contexts = torch.stack([s.mean(dim=(1, 2)) for s in upsampled], dim=1)

        # Learned weights
        weights = self.attn_mlp(scale_contexts).unsqueeze(-1).unsqueeze(-1)
        stacked = torch.stack(upsampled, dim=1)
        return torch.sum(stacked * weights, dim=1)


class DistributionAwareScoring(nn.Module):
    """Mahalanobis distance + spectral divergence scoring.
    More sensitive than MSE to distributional and frequency anomalies."""
    def __init__(self, seq_len):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(1))
        self.spectral_weight = nn.Parameter(torch.ones(seq_len // 2 + 1) * 0.1)

    def forward(self, reconstruction, ground_truth):
        """
        Returns per-element anomaly scores [B, T, C].
        """
        # Mahalanobis-like (variance-normalized)
        var = torch.exp(self.log_var) + 1e-6
        mahal_score = ((reconstruction - ground_truth) ** 2) / var

        # Spectral divergence
        recon_fft = torch.fft.rfft(reconstruction, dim=1)
        gt_fft = torch.fft.rfft(ground_truth, dim=1)
        spec_diff = (recon_fft - gt_fft).abs()
        F_bins = spec_diff.shape[1]

        sw = torch.sigmoid(self.spectral_weight)
        if sw.shape[0] != F_bins:
            sw = F.interpolate(sw.unsqueeze(0).unsqueeze(0), size=F_bins,
                             mode='linear', align_corners=False).squeeze(0).squeeze(0)
        weights = sw.unsqueeze(0).unsqueeze(-1)
        spec_score = (spec_diff * weights)
        spec_score = torch.fft.irfft(spec_score * torch.exp(1j * recon_fft.angle()),
                                     n=ground_truth.shape[1], dim=1).abs()

        return mahal_score + spec_score
