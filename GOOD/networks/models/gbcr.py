"""
Granular-Ball Cross-Reweight (GBCR) module.

Placed between GAT layers: takes mid-level node representations H,
forms soft granular balls, estimates ball-level importance,
and maps importance back to node/edge level as attention guidance.

This is NOT a separate branch. It acts as a mid-level importance estimator
within the GNN, providing "where to attend" hints for the next layer.

Design:
  Step 1: Soft assignment Q = softmax(W_q h / τ)  ->  [N, K]
  Step 2: Ball representation B = D_Q^{-1} Q^T H  ->  [K, d]
  Step 3: Ball node importance r = σ(MLP(B))  ->  [K]
           Ball edge importance S = σ(B W_s B^T)  ->  [K, K]
  Step 4: Node reweight u = Qr, h+ = h ⊙ (1 + α_n u)
           Edge reweight M_ij = (QSQ^T)[i,j], used as attention bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GBCR(nn.Module):
    """Granular-Ball Cross-Reweight module.

    Args:
        feat_dim: dimension of mid-level node features (dim_hidden)
        num_balls: number of granular balls K (12-16 for 100 ROI)
        alpha_node: residual scaling for node reweight (0.1-0.2)
        tau: temperature for soft assignment (lower = harder)
    """

    def __init__(self, feat_dim: int, num_balls: int = 14,
                 alpha_node: float = 0.15,
                 tau: float = 1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_balls = num_balls
        self.alpha_node = alpha_node
        self.tau = tau

        # Step 1: Soft assignment projection
        self.W_q = nn.Linear(feat_dim, num_balls, bias=False)

        # Step 3a: Ball node importance MLP
        self.ball_importance = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 1),
        )

        # Stored for loss computation
        self._last_Q = None
        self._last_r = None

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> tuple:
        """
        Args:
            H: [N_total, feat_dim] mid-level node features (after GAT layer 1)
            edge_index: [2, E] edge indices
            batch: [N_total] batch indicator

        Returns:
            H_reweighted: [N_total, feat_dim] node features with importance reweighting
            edge_importance: [E] per-edge importance bias for next GAT layer
            gbcr_info: dict with Q, r, S for loss computation
        """
        device = H.device
        num_graphs = int(batch.max().item()) + 1

        # ---- Step 1: Soft ball assignment ----
        # Q_ik = softmax(W_q h_i / tau)  per graph
        logits = self.W_q(H) / self.tau  # [N, K]
        # Mask so softmax is per-graph (nodes from different graphs don't share balls)
        Q = self._per_graph_softmax(logits, batch, num_graphs)  # [N, K]

        # ---- Step 2: Ball representations ----
        # B_k = (1/|ball_k|) sum_i Q_ik * h_i  for each graph
        # Vectorized across batch
        B = self._compute_ball_repr(H, Q, batch, num_graphs)  # [total_balls, feat_dim]
        # Ball-to-graph mapping
        ball_batch = torch.arange(num_graphs, device=device).repeat_interleave(self.num_balls)

        # ---- Step 3: Ball node importance ----
        r = torch.sigmoid(self.ball_importance(B).squeeze(-1))  # [total_balls]

        # ---- Step 4: Node reweight ----
        # u_i = sum_k Q_ik * r_k for the graph that node i belongs to.
        u = self._ball_to_node_importance(Q, r, batch, num_graphs)  # [N]
        H_reweighted = H * (1.0 + self.alpha_node * u.unsqueeze(-1))

        # NOTE: ball-ball bilinear edge importance (W_s, _compute_edge_importance)
        # was removed — it was computed every forward but never consumed by the
        # downstream GAT layer (see GAT.py: convs use the original edge_weight),
        # so it only inflated activation memory and contributed to OOM in
        # episodic mode. Re-add later if/when GAT2 actually consumes a
        # ball-derived edge bias.

        self._last_Q = Q
        self._last_r = r
        self._last_ball_batch = ball_batch

        gbcr_info = {
            'Q': Q,           # [N, K] soft assignment
            'r': r,           # [total_balls] ball importance
            'ball_batch': ball_batch,
        }

        return H_reweighted, gbcr_info

    def _per_graph_softmax(self, logits, batch, num_graphs):
        """Softmax over K balls, independently per graph."""
        # Standard softmax along dim=1 is already per-node over K balls
        # No cross-graph leaking since each node only accesses its own K balls
        return F.softmax(logits, dim=1)

    def _compute_ball_repr(self, H, Q, batch, num_graphs):
        """Compute ball representations B = D_Q^{-1} Q^T H per graph.

        Returns: [num_graphs * K, feat_dim]
        """
        K = self.num_balls
        d = self.feat_dim
        device = H.device

        # Offset Q columns per graph so scatter works globally
        # ball_idx[i, k] = graph_of_i * K + k
        graph_offset = batch.unsqueeze(1) * K  # [N, 1]
        # For each node i and ball k, accumulate Q_ik * h_i into ball (graph*K + k)
        total_balls = num_graphs * K

        # Weighted sum: B[ball] += Q[i, k] * H[i]
        B = torch.zeros(total_balls, d, device=device)
        D = torch.zeros(total_balls, 1, device=device)  # normalizer

        for k in range(K):
            ball_idx = graph_offset.squeeze(1) + k  # [N]
            weight = Q[:, k].unsqueeze(1)  # [N, 1]
            B.scatter_add_(0, ball_idx.unsqueeze(1).expand(-1, d), weight * H)
            D.scatter_add_(0, ball_idx.unsqueeze(1), weight)

        B = B / D.clamp(min=1e-6)
        return B

    def _ball_to_node_importance(self, Q, r, batch, num_graphs):
        """Map ball importance r back to nodes: u_i = sum_k Q_ik * r_k.

        Returns: [N]
        """
        K = self.num_balls
        # r is [total_balls], reshape to [num_graphs, K]
        r_3d = r.view(num_graphs, K)
        # For each node i in graph g: u_i = sum_k Q_ik * r_3d[g, k]
        r_per_node = r_3d[batch]  # [N, K]
        u = (Q * r_per_node).sum(dim=1)  # [N]
        return u

    def importance_alignment_loss(self, batch: torch.Tensor,
                                  labels: torch.Tensor,
                                  env_ids: torch.Tensor) -> torch.Tensor:
        """Class-conditional ball importance alignment across sites.

        L_gb-dg = sum_c sum_s ||r_bar_{s,c} - r_bar_c||^2

        For each class c, across source sites, the ball importance pattern
        should be similar — same disease should highlight similar brain modules.

        Args:
            batch: [N] batch indicator (not used directly, but stored Q/r have it)
            labels: [B] class labels per graph
            env_ids: [B] site/environment IDs per graph

        Returns:
            alignment loss (scalar)
        """
        if self._last_r is None:
            return torch.tensor(0.0, device=batch.device)

        r = self._last_r  # [num_graphs * K]
        K = self.num_balls
        num_graphs = r.size(0) // K
        r_per_graph = r.view(num_graphs, K)  # [B, K]

        labels = labels.view(-1)
        env_ids = env_ids.view(-1)

        unique_classes = labels.unique()
        unique_envs = env_ids.unique()

        if unique_envs.size(0) <= 1 or unique_classes.size(0) <= 1:
            return torch.tensor(0.0, device=r.device)

        loss = torch.tensor(0.0, device=r.device)
        for c in unique_classes:
            class_mask = labels == c
            if class_mask.sum() < 2:
                continue
            r_class = r_per_graph[class_mask]  # [n_c, K]
            r_bar_c = r_class.mean(dim=0)  # [K] global class mean

            for s in unique_envs:
                site_class_mask = class_mask & (env_ids == s)
                if site_class_mask.sum() == 0:
                    continue
                r_bar_sc = r_per_graph[site_class_mask].mean(dim=0)  # [K]
                loss = loss + (r_bar_sc - r_bar_c).pow(2).sum()

        # Normalize by number of (class, site) pairs
        n_pairs = max(unique_classes.size(0) * unique_envs.size(0), 1)
        return loss / n_pairs

    def assignment_entropy_loss(self) -> torch.Tensor:
        """Encourage non-degenerate assignment (each ball gets some nodes).

        Uses entropy of column sums of Q: higher entropy = more balanced.
        We minimize negative entropy (maximize entropy).
        """
        if self._last_Q is None:
            return torch.tensor(0.0)

        Q = self._last_Q  # [N, K]
        # Column sum = how many nodes each ball attracts
        col_sum = Q.sum(dim=0)  # [K]
        col_dist = col_sum / col_sum.sum().clamp(min=1e-8)
        col_dist = col_dist.clamp(min=1e-8)
        entropy = -(col_dist * col_dist.log()).sum()
        # Maximize entropy = minimize -entropy
        max_entropy = torch.log(torch.tensor(float(self.num_balls), device=Q.device))
        return (max_entropy - entropy) / max_entropy  # normalized to [0, 1]
