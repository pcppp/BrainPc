from types import SimpleNamespace

import torch
from torch_geometric.data import Data


def _graph_label_to_int(y: torch.Tensor) -> int:
    if y is None:
        return 0
    if y.dim() == 0:
        return int(y.item())
    if y.dim() == 1:
        if y.numel() == 1:
            return int(y.item())
        return int(y.argmax().item())
    return int(y.view(y.size(0), -1)[0].argmax(dim=-1).item())


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


def build_granular_ball_view(data: Data, ball_r: float = 0.5) -> Data:
    """Build a coarse PyG graph view using granular-ball division.

    The returned graph keeps the original feature dimension so it can serve as a
    semantically aligned positive view for cross-scale contrastive learning.
    """
    if data.x is None or data.x.size(0) <= 1 or data.edge_index is None or data.edge_index.numel() == 0:
        return _fallback_graph_view(data)

    try:
        from GOOD.data.gnn_nodesclassify_back.gb_division import gb_division
    except Exception as exc:
        print(f"⚠️ Failed to import granular-ball builder, fallback to original view: {exc}")
        return _fallback_graph_view(data)

    graph_label = _graph_label_to_int(getattr(data, 'y', None))
    num_nodes = int(data.x.size(0))
    work_data = Data(
        x=data.x.detach().cpu().float(),
        edge_index=data.edge_index.detach().cpu().long(),
        y=torch.full((num_nodes,), graph_label, dtype=torch.long),
        test_mask=torch.zeros(num_nodes, dtype=torch.bool),
    )

    args = SimpleNamespace(ball_r=ball_r, noisy=0)

    try:
        gb_result, _, _ = gb_division(work_data, args)
    except Exception as exc:
        print(f"⚠️ Granular-ball division failed, fallback to original view: {exc}")
        return _fallback_graph_view(data)

    gb_features = torch.as_tensor(gb_result.get('gb_features'), dtype=torch.float32)
    gb_adj = torch.as_tensor(gb_result.get('adj'), dtype=torch.long)

    if gb_features.ndim == 1:
        gb_features = gb_features.unsqueeze(0)

    if gb_features.numel() == 0 or gb_features.size(0) == 0:
        return _fallback_graph_view(data)

    num_balls = int(gb_features.size(0))
    if gb_adj.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = gb_adj.view(2, -1).long()

    if edge_index.numel() > 0:
        edge_weight = torch.ones(edge_index.size(1), 1, dtype=torch.float32)
    else:
        edge_weight = torch.empty((0, 1), dtype=torch.float32)

    coarse = Data(
        x=gb_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=data.y.clone() if getattr(data, 'y', None) is not None else None,
    )
    coarse.edge_attr = edge_weight
    coarse.num_nodes = num_balls

    if hasattr(data, 'domain'):
        coarse.domain = data.domain
    if hasattr(data, 'env_id'):
        coarse.env_id = data.env_id

    return coarse
