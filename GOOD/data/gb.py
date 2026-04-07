from collections import defaultdict

import torch
from torch_geometric.data import Data


def _fallback_graph_view(data: Data) -> Data:
    view = data.clone()
    view.x = data.x.clone().float()
    view.edge_index = data.edge_index.clone().long()
    if getattr(data, 'edge_weight', None) is not None:
        view.edge_weight = data.edge_weight.clone().float()
    if getattr(data, 'edge_attr', None) is not None:
        view.edge_attr = data.edge_attr.clone().float()
    if getattr(data, 'y', None) is not None:
        view.y = data.y.clone()
    if hasattr(data, 'domain'):
        view.domain = data.domain
    if hasattr(data, 'env_id'):
        view.env_id = data.env_id
    return view


def _zscore(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (x - mean) / std


def _build_structural_feature(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.index_add_(0, src, edge_weight.abs())
    degree.index_add_(0, dst, edge_weight.abs())

    binary_degree = torch.zeros(num_nodes, dtype=torch.float32)
    ones = torch.ones_like(src, dtype=torch.float32)
    binary_degree.index_add_(0, src, ones)
    binary_degree.index_add_(0, dst, ones)

    degree = degree.unsqueeze(-1)
    binary_degree = binary_degree.unsqueeze(-1)
    return torch.cat([_zscore(degree), _zscore(binary_degree)], dim=-1)


def _select_seed_indices(node_repr: torch.Tensor, seed_count: int, degree_score: torch.Tensor) -> torch.Tensor:
    num_nodes = node_repr.size(0)
    seed_count = max(2, min(seed_count, num_nodes))

    first_seed = int(torch.argmax(degree_score).item())
    selected = [first_seed]
    min_dist = torch.cdist(node_repr[first_seed:first_seed + 1], node_repr).squeeze(0)

    for _ in range(1, seed_count):
        min_dist[selected] = -1.0
        next_seed = int(torch.argmax(min_dist).item())
        selected.append(next_seed)
        next_dist = torch.cdist(node_repr[next_seed:next_seed + 1], node_repr).squeeze(0)
        min_dist = torch.minimum(min_dist.clamp_min(0), next_dist)

    return torch.tensor(selected, dtype=torch.long)


def _build_hop_distance(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    inf = num_nodes + 1
    hop = torch.full((num_nodes, num_nodes), float(inf), dtype=torch.float32)
    hop.fill_diagonal_(0.0)
    src, dst = edge_index
    hop[src, dst] = 1.0
    hop[dst, src] = 1.0

    # Floyd-Warshall is acceptable here because ABIDE graphs are only 100 nodes.
    for k in range(num_nodes):
        hop = torch.minimum(hop, hop[:, k:k + 1] + hop[k:k + 1, :])

    unreachable = hop > inf / 2
    if unreachable.any():
        hop[unreachable] = float(num_nodes)
    return hop


def _cluster_nodes(data: Data, ball_r: float) -> tuple[torch.Tensor, int]:
    num_nodes = int(data.x.size(0))
    target_balls = max(2, min(num_nodes - 1, int(round(max(ball_r, 0.05) * num_nodes))))

    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is None:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32, device=data.x.device)
    edge_weight = edge_weight.view(-1).float().cpu()
    edge_index = data.edge_index.long().cpu()
    x = data.x.float().cpu()

    structural = _build_structural_feature(edge_index, edge_weight, num_nodes)
    node_repr = torch.cat([_zscore(x), 0.5 * structural], dim=-1)

    seed_indices = _select_seed_indices(node_repr, target_balls, structural[:, 0])
    feature_dist = torch.cdist(node_repr, node_repr[seed_indices])
    hop = _build_hop_distance(edge_index, num_nodes)
    hop_dist = hop[:, seed_indices] / max(float(num_nodes), 1.0)
    cluster_assign = torch.argmin(feature_dist + 0.35 * hop_dist, dim=1)
    return cluster_assign, target_balls


def _aggregate_coarse_graph(data: Data, cluster_assign: torch.Tensor, target_balls: int) -> Data:
    x = data.x.float().cpu()
    num_nodes = int(x.size(0))
    num_balls = int(cluster_assign.max().item()) + 1
    num_balls = max(1, min(num_balls, target_balls))

    coarse_x = torch.zeros(num_balls, x.size(1), dtype=torch.float32)
    counts = torch.bincount(cluster_assign, minlength=num_balls).float().clamp_min(1.0)
    coarse_x.index_add_(0, cluster_assign, x)
    coarse_x = coarse_x / counts.unsqueeze(-1)

    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is None:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32)
    edge_weight = edge_weight.view(-1).float().cpu()
    edge_index = data.edge_index.long().cpu()

    pair_sum = defaultdict(float)
    pair_count = defaultdict(int)
    src_cluster = cluster_assign[edge_index[0]]
    dst_cluster = cluster_assign[edge_index[1]]
    for idx in range(edge_index.size(1)):
        u = int(src_cluster[idx].item())
        v = int(dst_cluster[idx].item())
        if u == v:
            continue
        key = (u, v)
        pair_sum[key] += float(edge_weight[idx].item())
        pair_count[key] += 1

    if pair_sum:
        edge_pairs = sorted(pair_sum.keys())
        coarse_edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        coarse_edge_weight = torch.tensor([
            pair_sum[key] / max(pair_count[key], 1) for key in edge_pairs
        ], dtype=torch.float32).unsqueeze(-1)
    else:
        coarse_edge_index = torch.empty((2, 0), dtype=torch.long)
        coarse_edge_weight = torch.empty((0, 1), dtype=torch.float32)

    coarse = Data(
        x=coarse_x,
        edge_index=coarse_edge_index,
        edge_weight=coarse_edge_weight,
        y=data.y.clone() if getattr(data, 'y', None) is not None else None,
    )
    coarse.edge_attr = coarse_edge_weight
    coarse.num_nodes = num_balls
    if hasattr(data, 'domain'):
        coarse.domain = data.domain
    if hasattr(data, 'env_id'):
        coarse.env_id = data.env_id
    return coarse


def build_granular_ball_view(data: Data, ball_r: float = 0.5) -> Data:
    """Build a coarse graph view with unsupervised feature-structure granular-ball assignment.

    The previous graph-classification path reused a node-classification granular-ball splitter
    that relied on node labels and purity. In this project each graph only has a graph label,
    so copying that label to every node makes purity-based splitting degenerate. The current
    implementation therefore uses unsupervised seed selection and assignment based on both
    node features and graph-hop structure, then preserves aggregated inter-ball edge weights.
    """
    if data.x is None or data.x.size(0) <= 2 or data.edge_index is None or data.edge_index.numel() == 0:
        return _fallback_graph_view(data)

    try:
        cluster_assign, target_balls = _cluster_nodes(data, ball_r=ball_r)
        coarse = _aggregate_coarse_graph(data, cluster_assign, target_balls)
    except Exception as exc:
        print(f"⚠️ Granular-ball view fallback to original view: {exc}")
        return _fallback_graph_view(data)

    if coarse.x.numel() == 0 or coarse.x.size(0) <= 1:
        return _fallback_graph_view(data)
    return coarse
