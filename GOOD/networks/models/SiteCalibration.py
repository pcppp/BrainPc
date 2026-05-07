"""
Sample-level statistical-aware site calibration module.
Placed between GAT layer 1 and layer 2 (mid-level representations).

Design (v3, revived):
  - Higher effective magnitude (alpha=0.25, scale_bound=0.2)
  - Gate floor (min 0.03) prevents full shutdown
  - Range penalty (target gate mean in [0.03, 0.15]) instead of push-to-zero
  - LayerNorm on stats input so edge stats can actually influence gate
  - Small random init on final weight (1e-3) for faster warmup
  - Larger bottleneck (feat_dim // 4, min 16)
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
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
    """Sample-level residual site calibration via meta-network (v3).

    Args:
        feat_dim: dimension of node features (= dim_hidden after GAT1)
        gate_init_bias: initial bias for gate logit (-1.7 => sigmoid ≈ 0.154, with floor=0.01 -> init ≈ 0.162)
        num_sites: number of sites for adversarial classifier
        alpha: residual scaling factor
        scale_bound: bound for tanh scaling of gamma/beta
        gate_floor: minimum gate value after sigmoid
        gate_range: target (min, max) for L_gate_range penalty
    """

    def __init__(self, feat_dim: int, gate_init_bias: float = -1.7,
                 num_sites: int = 0, alpha: float = 0.25,
                 scale_bound: float = 0.2, gate_floor: float = 0.01,
                 gate_range: tuple = (0.05, 0.15)):
        super().__init__()
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.scale_bound = scale_bound
        self.gate_floor = gate_floor
        self.gate_min, self.gate_max = gate_range

        # Meta-network input: 2*feat_dim (H stats) + 4 (edge stats)
        stat_dim = 2 * feat_dim + 4
        bottleneck = max(feat_dim // 4, 16)

        # LayerNorm so edge stats have comparable magnitude to H stats
        self.stats_norm = nn.LayerNorm(stat_dim)

        self.meta_net = nn.Sequential(
            nn.Linear(stat_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, 3 * feat_dim),
        )

        # Small random init (not zero) for final weight; bias still identity
        with torch.no_grad():
            final = self.meta_net[-1]
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            final.bias[:feat_dim].zero_()             # gamma_hat = 0
            final.bias[feat_dim:2*feat_dim].zero_()   # beta_hat = 0
            final.bias[2*feat_dim:].fill_(gate_init_bias)

        # Gate EMA buffer for alignment loss (smooths across mini-batches).
        self._gate_ema = {}  # (class_id, site_id) -> running mean [feat_dim]
        self._gate_ema_momentum = 0.1

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
        """Per-graph edge stats: mean(|w|), std(|w|), density, pos_ratio.

        Vectorised via scatter_add_ — one pass over all edges instead of
        num_graphs Python iterations. Std uses the population (biased)
        estimator sqrt(E[w^2] - E[w]^2) so a single-edge graph yields 0
        without the unbiased NaN edge case.
        """
        edge_stats = torch.zeros(num_graphs, 4, device=device)
        if edge_weight is None or edge_weight.numel() == 0:
            return edge_stats

        # Normalise shapes: edge_weight may be [E] or [E, 1].
        w = edge_weight.view(-1).float()
        abs_w = w.abs()
        edge_batch = batch[edge_index[0]]            # [E] graph id per edge

        # --- Per-graph counters ---
        edge_count = torch.zeros(num_graphs, device=device)
        edge_count.scatter_add_(0, edge_batch, torch.ones_like(abs_w))

        node_count = torch.zeros(num_graphs, device=device)
        node_count.scatter_add_(0, batch, torch.ones(batch.size(0), device=device))

        # --- mean(|w|) and E[w^2] for population std ---
        sum_abs = torch.zeros(num_graphs, device=device)
        sum_abs.scatter_add_(0, edge_batch, abs_w)
        mean_abs = sum_abs / edge_count.clamp(min=1)

        sum_sq = torch.zeros(num_graphs, device=device)
        sum_sq.scatter_add_(0, edge_batch, abs_w.pow(2))
        var = (sum_sq / edge_count.clamp(min=1)) - mean_abs.pow(2)
        std_abs = var.clamp(min=0).sqrt()

        # --- Density = E_g / (N_g * (N_g - 1)) ---
        max_edges = node_count * (node_count - 1)
        density = edge_count / max_edges.clamp(min=1)

        # --- Positive-edge ratio ---
        pos_sum = torch.zeros(num_graphs, device=device)
        pos_sum.scatter_add_(0, edge_batch, (w > 0).float())
        pos_ratio = pos_sum / edge_count.clamp(min=1)

        edge_stats = torch.stack([mean_abs, std_abs, density, pos_ratio], dim=1)

        # Graphs with zero edges should report zeros, not NaN/inf.
        empty_mask = edge_count == 0
        if empty_mask.any():
            edge_stats[empty_mask] = 0.0
        return edge_stats

    def forward(self, H, batch, edge_index=None, edge_weight=None):
        device = H.device
        num_graphs = int(batch.max().item()) + 1

        # --- Per-graph node feature statistics ---
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

        # --- LayerNorm + meta-network ---
        stats = self.stats_norm(torch.cat([graph_mean, graph_std, edge_stats], dim=1))
        params = self.meta_net(stats)
        gamma_hat = params[:, :self.feat_dim]
        beta_hat = params[:, self.feat_dim:2*self.feat_dim]
        gate_logit = params[:, 2*self.feat_dim:]

        # --- Constrained parameters ---
        gamma = 1.0 + self.scale_bound * torch.tanh(gamma_hat)  # [1-0.2, 1+0.2]
        beta = self.scale_bound * torch.tanh(beta_hat)            # [-0.2, 0.2]
        # Gate with floor: min self.gate_floor
        gate = self.gate_floor + (1 - self.gate_floor) * torch.sigmoid(gate_logit)

        # --- Broadcast to node level ---
        gamma_n = gamma[batch]
        beta_n = beta[batch]
        gate_n = gate[batch]
        mean_n = graph_mean[batch]
        std_n = graph_std[batch]

        # --- Residual calibration ---
        H_norm = (H - mean_n) / (std_n + 1e-6)
        delta = gamma_n * H_norm + beta_n
        H_calibrated = H + self.alpha * gate_n * (delta - H)

        # --- Regularization values ---
        # Range penalty: want gate_mean in [gate_min, gate_max], else linearly penalize
        gate_mean_global = gate.mean()
        gate_range_reg = (torch.clamp(self.gate_min - gate_mean_global, min=0)
                         + torch.clamp(gate_mean_global - self.gate_max, min=0))
        # L_aff still encourages identity when NOT in active use
        affine_reg = (gamma - 1).pow(2).mean() + beta.pow(2).mean()

        calib_info = {
            'gate_reg': gate_range_reg,     # now range penalty
            'affine_reg': affine_reg,
            'gamma': gamma,
            'beta': beta,
            'gate': gate,
            'gate_logit': gate_logit,
            'gate_mean': gate.mean().item(),
            'gate_std': gate.std().item(),
            'gamma_dev': (gamma - 1).pow(2).mean().sqrt().item(),
            'beta_norm': beta.pow(2).mean().sqrt().item(),
        }

        return H_calibrated, calib_info

    def gate_alignment_loss(self, gate, batch_graph, labels, env_ids):
        """Class-conditional gate alignment across sites (EMA-smoothed).

        Maintains a running average per (class, site) across mini-batches.
        Alignment loss pushes each site's current-batch gate toward the
        class-wide EMA mean.  Gradients flow through current-batch gate;
        the EMA reference is detached.
        """
        device = gate.device

        # --- Update EMA (no grad) ---
        with torch.no_grad():
            for c_val in labels.unique():
                c_mask = (labels == c_val)
                for s_val in env_ids[c_mask].unique():
                    sc_mask = c_mask & (env_ids == s_val)
                    if sc_mask.sum() == 0:
                        continue
                    g_cs = gate[sc_mask].mean(dim=0)
                    key = (c_val.item(), s_val.item())
                    if key not in self._gate_ema:
                        self._gate_ema[key] = g_cs.clone()
                    else:
                        m = self._gate_ema_momentum
                        self._gate_ema[key] = (1 - m) * self._gate_ema[key] + m * g_cs

        # --- Alignment loss from EMA ---
        loss = torch.tensor(0.0, device=device)
        count = 0
        for c_val in labels.unique():
            c_mask = (labels == c_val)
            # Class-mean gate across ALL sites (from EMA buffer)
            site_means = [(k, v) for k, v in self._gate_ema.items()
                          if k[0] == c_val.item()]
            if len(site_means) < 2:
                continue
            g_bar_c = torch.stack([v for _, v in site_means]).mean(dim=0).detach()

            # For each site in this batch: push toward class-wide mean
            for s_val in env_ids[c_mask].unique():
                sc_mask = c_mask & (env_ids == s_val)
                if sc_mask.sum() == 0:
                    continue
                g_bar_sc = gate[sc_mask].mean(dim=0)  # has gradient
                loss = loss + (g_bar_sc - g_bar_c).pow(2).mean()
                count += 1

        return loss / max(count, 1)

    @torch.no_grad()
    def site_probe_accuracy(self, H_pre, H_post, batch, site_labels):
        """Measure if gate is removing site info.

        Returns (pre_acc, post_acc). If post_acc << pre_acc, gate is working.
        """
        if self.site_classifier is None:
            return 0.0, 0.0

        num_graphs = int(batch.max().item()) + 1
        device = H_pre.device

        def pool(H):
            g = torch.zeros(num_graphs, self.feat_dim, device=device)
            cnt = torch.zeros(num_graphs, 1, device=device)
            cnt.index_add_(0, batch, torch.ones(H.size(0), 1, device=device))
            g.index_add_(0, batch, H)
            return g / cnt.clamp(min=1)

        g_pre = pool(H_pre.detach())
        g_post = pool(H_post.detach())

        logits_pre = self.site_classifier(g_pre)
        logits_post = self.site_classifier(g_post)

        pred_pre = logits_pre.argmax(dim=1)
        pred_post = logits_post.argmax(dim=1)

        acc_pre = (pred_pre == site_labels.long()).float().mean().item()
        acc_post = (pred_post == site_labels.long()).float().mean().item()
        return acc_pre, acc_post

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
