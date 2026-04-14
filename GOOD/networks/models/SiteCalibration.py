"""
Sample-level statistical-aware site calibration module.

Residual affine modulation with meta-network generated parameters.
Suppresses site/scanner style shift while preserving disease-relevant signals.

Design:
  - Extract per-sample statistics s = [mean(H), std(H)]  (stop-gradient)
  - Small meta-network: s -> (gamma, beta, gate)  channel-wise
  - Sample-internal standardization: H_norm = (H - mu) / (sigma + eps)
  - Residual calibration: H' = H + gate * (gamma * H_norm + beta - H)
  - gate initialized near 0 (identity by default)
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    """Gradient Reversal Layer for adversarial training."""
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x, lam=1.0):
    return _GradReverse.apply(x, lam)


class SiteCalibration(nn.Module):
    """Sample-level residual site calibration via meta-network affine modulation.

    Only generates channel-wise parameters to avoid overfitting on small datasets.
    Bottleneck = d/8 for small-sample regime.

    Args:
        feat_dim: dimension of node features
        gate_init_bias: initial bias for gate (negative = near-zero gate = near-identity)
        num_sites: number of sites for adversarial classifier (0 = disable)
    """

    def __init__(self, feat_dim: int, gate_init_bias: float = -3.0, num_sites: int = 0):
        super().__init__()
        self.feat_dim = feat_dim

        # Meta-network: 2d -> d/8 -> 3d  (very small for small-sample)
        bottleneck = max(feat_dim // 8, 8)
        self.meta_net = nn.Sequential(
            nn.Linear(2 * feat_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, 3 * feat_dim),
        )

        # Initialize: gamma~1, beta~0, gate~sigmoid(gate_init_bias)~0.05
        with torch.no_grad():
            final_layer = self.meta_net[-1]
            final_layer.weight.zero_()
            final_layer.bias[:feat_dim].fill_(1.0)       # gamma = 1
            final_layer.bias[feat_dim:2*feat_dim].zero_() # beta = 0
            final_layer.bias[2*feat_dim:].fill_(gate_init_bias)  # gate near 0

        # Site adversarial classifier (gradient reversal)
        if num_sites > 0:
            self.site_classifier = nn.Sequential(
                nn.Linear(feat_dim, max(feat_dim // 4, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(max(feat_dim // 4, 8), num_sites),
            )
        else:
            self.site_classifier = None

    def forward(self, H: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            H: node features [N_total, feat_dim]
            batch: batch indicator [N_total]

        Returns:
            H_calibrated: [N_total, feat_dim]
            calib_info: dict with gate_reg, affine_reg, gamma, beta, gate
        """
        num_graphs = int(batch.max().item()) + 1
        graph_mean = torch.zeros(num_graphs, self.feat_dim, device=H.device)
        graph_count = torch.zeros(num_graphs, 1, device=H.device)

        graph_count.index_add_(0, batch, torch.ones(H.size(0), 1, device=H.device))
        graph_mean.index_add_(0, batch, H.detach())  # stop-gradient
        graph_mean = graph_mean / graph_count.clamp(min=1)

        graph_sq = torch.zeros(num_graphs, self.feat_dim, device=H.device)
        graph_sq.index_add_(0, batch, (H.detach()) ** 2)
        graph_sq = graph_sq / graph_count.clamp(min=1)
        graph_var = (graph_sq - graph_mean ** 2).clamp(min=1e-6)
        graph_std = graph_var.sqrt()

        stats = torch.cat([graph_mean, graph_std], dim=1)
        params = self.meta_net(stats)
        gamma = params[:, :self.feat_dim]
        beta = params[:, self.feat_dim:2*self.feat_dim]
        gate_logits = params[:, 2*self.feat_dim:]
        gate = torch.sigmoid(gate_logits)

        gamma_n = gamma[batch]
        beta_n = beta[batch]
        gate_n = gate[batch]
        mean_n = graph_mean[batch]
        std_n = graph_std[batch]

        H_norm = (H - mean_n) / (std_n + 1e-6)

        # H' = H + g * (gamma * H_norm + beta - H)
        H_calibrated = H + gate_n * (gamma_n * H_norm + beta_n - H)

        # L_sparse = ||g||_1
        gate_reg = gate.mean()
        # L_affine = ||gamma - 1||^2 + ||beta||^2  (encourage identity)
        affine_reg = (gamma - 1).pow(2).mean() + beta.pow(2).mean()

        calib_info = {
            'gate_reg': gate_reg,
            'affine_reg': affine_reg,
            'gamma': gamma,
            'beta': beta,
            'gate': gate,
        }

        return H_calibrated, calib_info

    def apply_params(self, H: torch.Tensor, batch: torch.Tensor,
                     gamma: torch.Tensor, beta: torch.Tensor, gate: torch.Tensor):
        """Apply pre-computed calibration params to a different view.

        Used to share the same calibration (estimated from original graph)
        with the granular-ball coarse view, avoiding scale-mismatch artifacts.

        Args:
            H: node features [N_total, feat_dim] of the target view
            batch: batch indicator [N_total]
            gamma, beta, gate: [num_graphs, feat_dim] from a prior forward() call
        """
        num_graphs = int(batch.max().item()) + 1

        # Per-node stats from the target view (for normalization only)
        graph_mean = torch.zeros(num_graphs, self.feat_dim, device=H.device)
        graph_count = torch.zeros(num_graphs, 1, device=H.device)
        graph_count.index_add_(0, batch, torch.ones(H.size(0), 1, device=H.device))
        graph_mean.index_add_(0, batch, H.detach())
        graph_mean = graph_mean / graph_count.clamp(min=1)

        graph_sq = torch.zeros(num_graphs, self.feat_dim, device=H.device)
        graph_sq.index_add_(0, batch, (H.detach()) ** 2)
        graph_sq = graph_sq / graph_count.clamp(min=1)
        graph_std = (graph_sq - graph_mean ** 2).clamp(min=1e-6).sqrt()

        # Broadcast to node level
        gamma_n = gamma[batch]
        beta_n = beta[batch]
        gate_n = gate[batch]
        mean_n = graph_mean[batch]
        std_n = graph_std[batch]

        H_norm = (H - mean_n) / (std_n + 1e-6)
        H_calibrated = H + gate_n * (gamma_n * H_norm + beta_n - H)
        return H_calibrated

    def site_adversarial_loss(self, H_calibrated, batch, site_labels, grl_lambda=1.0):
        """Site adversarial loss on calibrated features (gradient reversal)."""
        if self.site_classifier is None:
            return torch.tensor(0.0, device=H_calibrated.device)

        num_graphs = int(batch.max().item()) + 1
        graph_feat = torch.zeros(num_graphs, self.feat_dim, device=H_calibrated.device)
        count = torch.zeros(num_graphs, 1, device=H_calibrated.device)
        count.index_add_(0, batch, torch.ones(H_calibrated.size(0), 1, device=H_calibrated.device))
        graph_feat.index_add_(0, batch, H_calibrated)
        graph_feat = graph_feat / count.clamp(min=1)

        graph_feat_rev = grad_reverse(graph_feat, grl_lambda)
        site_logits = self.site_classifier(graph_feat_rev)
        loss = torch.nn.functional.cross_entropy(site_logits, site_labels.long())
        return loss

    @staticmethod
    def gate_consistency_loss(calib_info_1, calib_info_2):
        """Same sample two views should have similar gate params."""
        loss = (
            (calib_info_1['gate'] - calib_info_2['gate']).abs().mean() +
            (calib_info_1['gamma'] - calib_info_2['gamma']).abs().mean() +
            (calib_info_1['beta'] - calib_info_2['beta']).abs().mean()
        )
        return loss
