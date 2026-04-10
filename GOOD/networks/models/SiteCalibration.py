"""
Sample-level statistical-aware site calibration module.

Residual affine modulation with meta-network generated parameters.
Suppresses site/scanner style shift while preserving disease-relevant signals.

Design:
  - Extract per-sample statistics s = [mean(H), std(H)]  (stop-gradient)
  - Small meta-network: s -> (alpha, beta, gate)  channel-wise
  - Calibration: H' = H + gate * (alpha * H_norm + beta - H)
  - gate initialized near 0 (identity by default)
"""

import torch
import torch.nn as nn


class SiteCalibration(nn.Module):
    """Sample-level residual site calibration via meta-network affine modulation.

    Placed between shallow feature extractor (CNN/LSTM) and GNN encoder.
    Only generates channel-wise parameters to avoid overfitting on small datasets.

    Args:
        feat_dim: dimension of node features (e.g. lstm_hidden_size=128)
        gate_init_bias: initial bias for gate (negative = near-zero gate = near-identity)
    """

    def __init__(self, feat_dim: int, gate_init_bias: float = -3.0):
        super().__init__()
        self.feat_dim = feat_dim

        # Meta-network: 2*feat_dim (mean+std) -> alpha, beta, gate (3*feat_dim)
        # Deliberately small: one hidden layer, bottleneck = feat_dim//4
        bottleneck = max(feat_dim // 4, 16)
        self.meta_net = nn.Sequential(
            nn.Linear(2 * feat_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, 3 * feat_dim),
        )

        # Initialize: alpha~1, beta~0, gate~sigmoid(gate_init_bias)~0.05
        with torch.no_grad():
            final_layer = self.meta_net[-1]
            final_layer.weight.zero_()
            # alpha bias = 1.0 (identity scaling)
            final_layer.bias[:feat_dim].fill_(1.0)
            # beta bias = 0.0
            final_layer.bias[feat_dim:2*feat_dim].zero_()
            # gate bias = gate_init_bias (small positive after sigmoid)
            final_layer.bias[2*feat_dim:].fill_(gate_init_bias)

    def forward(self, H: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: node features [N_total, feat_dim] (all nodes in batch)
            batch: batch indicator [N_total] mapping each node to its graph index

        Returns:
            H_calibrated: [N_total, feat_dim]
            gate_reg: scalar, mean gate magnitude for sparsity regularization
        """
        # --- Per-graph statistics (stop-gradient) ---
        num_graphs = int(batch.max().item()) + 1
        graph_mean = torch.zeros(num_graphs, self.feat_dim, device=H.device)
        graph_std = torch.zeros(num_graphs, self.feat_dim, device=H.device)
        graph_count = torch.zeros(num_graphs, 1, device=H.device)

        # Scatter-based per-graph statistics
        graph_count.index_add_(0, batch, torch.ones(H.size(0), 1, device=H.device))
        graph_mean.index_add_(0, batch, H.detach())  # stop-gradient on stats
        graph_mean = graph_mean / graph_count.clamp(min=1)

        # Variance via E[X^2] - E[X]^2
        graph_sq = torch.zeros(num_graphs, self.feat_dim, device=H.device)
        graph_sq.index_add_(0, batch, (H.detach()) ** 2)
        graph_sq = graph_sq / graph_count.clamp(min=1)
        graph_var = (graph_sq - graph_mean ** 2).clamp(min=1e-6)
        graph_std = graph_var.sqrt()

        # --- Meta-network generates channel-wise alpha, beta, gate ---
        stats = torch.cat([graph_mean, graph_std], dim=1)  # [B, 2*feat_dim]
        params = self.meta_net(stats)  # [B, 3*feat_dim]
        alpha = params[:, :self.feat_dim]                   # [B, feat_dim]
        beta = params[:, self.feat_dim:2*self.feat_dim]     # [B, feat_dim]
        gate_logits = params[:, 2*self.feat_dim:]           # [B, feat_dim]
        gate = torch.sigmoid(gate_logits)                   # [B, feat_dim] in (0,1)

        # --- Broadcast to node level ---
        alpha_n = alpha[batch]    # [N_total, feat_dim]
        beta_n = beta[batch]
        gate_n = gate[batch]
        mean_n = graph_mean[batch]
        std_n = graph_std[batch]

        # --- Sample-internal standardization ---
        H_norm = (H - mean_n) / (std_n + 1e-6)

        # --- Residual calibration ---
        # H' = H + gate * (alpha * H_norm + beta - H)
        # gate=0 -> H (identity), gate=1 -> alpha*H_norm+beta (full affine norm)
        H_calibrated = H + gate_n * (alpha_n * H_norm + beta_n - H)

        # Gate magnitude for sparsity regularization
        gate_reg = gate.mean()

        return H_calibrated, gate_reg
