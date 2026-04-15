"""
Sample-level statistical-aware site calibration module.
Placed between GAT layer 1 and layer 2 (mid-level representations).

Design changes (v2):
  - Input: per-graph stats from both H (node features) AND edge weights
    s = [mean(H), std(H), mean(|A|), std(|A|), density, pos_ratio]
  - Constrained affine: gamma = 1 + 0.1*tanh(gamma_hat), beta = 0.1*tanh(beta_hat)
  - Residual form: H' = H + alpha * g * (gamma * H_norm + beta - H)
  - alpha fixed at 0.1 to limit calibration magnitude
  - Class-conditional gate alignment loss for domain generalization
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
    """Sample-level residual site calibration via meta-network.

    v2: constrained form, edge stats input, placed after GAT layer 1.

    Args:
        feat_dim: dimension of node features (= dim_hidden after GAT1)
        gate_init_bias: initial bias for gate logit (negative = near-zero gate)
        num_sites: number of sites for adversarial classifier (0 = disable)
        alpha: residual scaling factor (fixed, not learned)
        scale_bound: bound for tanh scaling of gamma/beta
    """

    def __init__(self, feat_dim: int, gate_init_bias: float = -3.0,
                 num_sites: int = 0, alpha: float = 0.1, scale_bound: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.scale_bound = scale_bound

        # Meta-network input: 2*feat_dim (H stats) + 4 (edge stats)
        # Edge stats: mean(|edge_weight|), std(|edge_weight|), density, positive_edge_ratio
        stat_dim = 2 * feat_dim + 4
        bottleneck = max(feat_dim // 8, 8)
        self.meta_net = nn.Sequential(
            nn.Linear(stat_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, 3 * feat_dim),  # gamma_hat, beta_hat, gate_logit
        )

        # Initialize: gamma_hat~0 (so gamma=1), beta_hat~0, gate near 0
        with torch.no_grad():
            final = self.meta_net[-1]
            final.weight.zero_()
            final.bias[:feat_dim].zero_()             # gamma_hat = 0 -> gamma = 1
            final.bias[feat_dim:2*feat_dim].zero_()   # beta_hat = 0 -> beta = 0
            final.bias[2*feat_dim:].fill_(gate_init_bias)  # gate near 0

        # Site adversarial classifier (gradient reversal)
        if num_sites > 0:
            self.site_classifier = nn.Sequential(
                nn.Linear(feat_dim, max(feat_dim // 4, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(max(feat_dim // 4, 8), num_sites),
            )
        else:
            self.site_classifier = None

    def _compute_edge_stats(self, edge_weight, edge_index, batch, num_graphs, device):
        """Compute per-graph edge statistics: mean(|w|), std(|w|), density, pos_ratio."""
        edge_stats = torch.zeros(num_graphs, 4, device=device)

        if edge_weight is None or edge_weight.numel() == 0:
            return edge_stats

        abs_w = edge_weight.abs()
        # Assign each edge to its source node's graph
        edge_batch = batch[edge_index[0]]

        for g in range(num_graphs):
            mask = (edge_batch == g)
            if mask.sum() == 0:
                continue
            w_g = abs_w[mask]
            n_nodes = (batch == g).sum().float()
            n_edges = mask.sum().float()

            edge_stats[g, 0] = w_g.mean()                           # mean(|A|)
            edge_stats[g, 1] = w_g.std() if w_g.numel() > 1 else 0  # std(|A|)
            max_edges = n_nodes * (n_nodes - 1)  # directed graph max
            edge_stats[g, 2] = n_edges / max_edges.clamp(min=1)      # density
            edge_stats[g, 3] = (edge_weight[mask] > 0).float().mean() # pos_ratio

        return edge_stats

    def forward(self, H, batch, edge_index=None, edge_weight=None):
        """
        Args:
            H: node features [N_total, feat_dim] (after GAT layer 1)
            batch: batch indicator [N_total]
            edge_index: [2, E] edge indices
            edge_weight: [E] edge weights (can be None)

        Returns:
            H_calibrated: [N_total, feat_dim]
            calib_info: dict with monitoring values
        """
        device = H.device
        num_graphs = int(batch.max().item()) + 1

        # --- Per-graph node feature statistics (stop-gradient) ---
        graph_mean = torch.zeros(num_graphs, self.feat_dim, device=device)
        graph_count = torch.zeros(num_graphs, 1, device=device)
        graph_count.index_add_(0, batch, torch.ones(H.size(0), 1, device=device))
        graph_mean.index_add_(0, batch, H.detach())
        graph_mean = graph_mean / graph_count.clamp(min=1)

        graph_sq = torch.zeros(num_graphs, self.feat_dim, device=device)
        graph_sq.index_add_(0, batch, (H.detach()) ** 2)
        graph_sq = graph_sq / graph_count.clamp(min=1)
        graph_var = (graph_sq - graph_mean ** 2).clamp(min=1e-6)
        graph_std = graph_var.sqrt()

        # --- Edge statistics ---
        edge_stats = self._compute_edge_stats(edge_weight, edge_index, batch, num_graphs, device)

        # --- Meta-network: [mean(H), std(H), edge_stats] -> (gamma_hat, beta_hat, gate_logit) ---
        stats = torch.cat([graph_mean, graph_std, edge_stats], dim=1)
        params = self.meta_net(stats)
        gamma_hat = params[:, :self.feat_dim]
        beta_hat = params[:, self.feat_dim:2*self.feat_dim]
        gate_logit = params[:, 2*self.feat_dim:]

        # --- Constrained parameters ---
        gamma = 1.0 + self.scale_bound * torch.tanh(gamma_hat)  # [1-0.1, 1+0.1]
        beta = self.scale_bound * torch.tanh(beta_hat)            # [-0.1, 0.1]
        gate = torch.sigmoid(gate_logit)                          # [0, 1]

        # --- Broadcast to node level ---
        gamma_n = gamma[batch]
        beta_n = beta[batch]
        gate_n = gate[batch]
        mean_n = graph_mean[batch]
        std_n = graph_std[batch]

        # --- Sample-internal standardization ---
        H_norm = (H - mean_n) / (std_n + 1e-6)

        # --- Residual calibration: H' = H + alpha * g * (gamma * H_norm + beta - H) ---
        H_calibrated = H + self.alpha * gate_n * (gamma_n * H_norm + beta_n - H)

        # --- Monitoring values ---
        gate_reg = gate.mean()  # L_sp = ||g||_1
        affine_reg = (gamma - 1).pow(2).mean() + beta.pow(2).mean()  # L_aff

        calib_info = {
            'gate_reg': gate_reg,
            'affine_reg': affine_reg,
            'gamma': gamma,        # [num_graphs, feat_dim]
            'beta': beta,
            'gate': gate,
            'gate_logit': gate_logit,
            # Scalar monitoring values
            'gate_mean': gate.mean().item(),
            'gate_std': gate.std().item(),
            'gamma_dev': (gamma - 1).pow(2).mean().sqrt().item(),  # ||gamma-1||
            'beta_norm': beta.pow(2).mean().sqrt().item(),         # ||beta||
        }

        return H_calibrated, calib_info

    def gate_alignment_loss(self, gate, batch_graph, labels, env_ids):
        """Class-conditional gate alignment across sites.

        L_gate-align = sum_c sum_s ||g_bar_{s,c} - g_bar_c||^2

        Same disease class across different sites should have similar gate patterns.

        Args:
            gate: [num_graphs, feat_dim] gate values
            batch_graph: not used (gate is already per-graph)
            labels: [num_graphs] class labels
            env_ids: [num_graphs] site/environment IDs
        """
        loss = torch.tensor(0.0, device=gate.device)
        classes = labels.unique()
        count = 0

        for c in classes:
            c_mask = (labels == c)
            if c_mask.sum() < 2:
                continue
            g_c = gate[c_mask]           # gates for class c
            g_bar_c = g_c.mean(dim=0)    # global mean for class c
            envs_c = env_ids[c_mask]

            for s in envs_c.unique():
                s_mask = (envs_c == s)
                if s_mask.sum() < 1:
                    continue
                g_bar_sc = g_c[s_mask].mean(dim=0)  # mean for site s, class c
                loss = loss + (g_bar_sc - g_bar_c).pow(2).mean()
                count += 1

        return loss / max(count, 1)

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
