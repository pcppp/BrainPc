import argparse
import os, re, glob
import numpy as np
import scipy.io
import torch
import dgl
import networkx as nx
from tqdm import tqdm
from dgl.data.utils import save_graphs

try:
    from GOOD.data.good_datasets.metadata_v5_utils import encode_binary_label, get_subject_row
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    _module_path = Path(__file__).resolve().parent / 'good_datasets' / 'metadata_v5_utils.py'
    _spec = importlib.util.spec_from_file_location('metadata_v5_utils', _module_path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    encode_binary_label = _module.encode_binary_label
    get_subject_row = _module.get_subject_row

BASEDIR = 'GOOD/data'
EPS = 1e-8


# ---------------------------------------------------------------------------------------------
def _key_from_path(p: str) -> str:
    """把文件名标准化成 subject key，用于 TS/FC 配对"""
    b = os.path.basename(p)
    b = re.sub(r'_schaefer100_(features_timeseries|correlation_matrix)\.mat$', '', b)
    return b

def _resolve_mat_key(mat_dict: dict, preferred_key: str = "data") -> str:
    if preferred_key in mat_dict:
        return preferred_key

    candidates = [
        (k, np.asarray(v))
        for k, v in mat_dict.items()
        if not k.startswith('__') and np.asarray(v).ndim == 2
    ]
    if not candidates:
        raise KeyError(f'No 2D array found in keys: {list(mat_dict.keys())}')

    for key in ('data', 'fc', 'features', 'timeseries'):
        for candidate_key, _ in candidates:
            if candidate_key == key:
                return candidate_key
    return max(candidates, key=lambda item: item[1].size)[0]


def _load_mat(path: str, key: str = "data") -> np.ndarray:
    mat = scipy.io.loadmat(path)
    actual_key = _resolve_mat_key(mat, preferred_key=key)
    array = np.asarray(mat[actual_key], dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f'{path} does not contain a 2D matrix under key {actual_key}: {array.shape}')
    if not np.isfinite(array).all():
        print(f'[WARN] {path} contains NaN/Inf values under key {actual_key}; replacing them with 0.')
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def _ensure_node_by_time(ts: np.ndarray) -> np.ndarray:
    if ts.ndim != 2:
        raise ValueError(f'Timeseries must be 2D, but got shape {ts.shape}.')
    if ts.shape[0] >= ts.shape[1]:
        return ts.T.copy()
    return ts.copy()


def _load_fc_matrix(path: str, num_nodes: int) -> np.ndarray:
    fc = _load_mat(path, key='data')
    if fc.shape != (num_nodes, num_nodes):
        raise ValueError(f'FC shape mismatch in {path}: expected {(num_nodes, num_nodes)}, got {fc.shape}')
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 0.0)
    return fc


def _sparsify_fc_matrix(fc_matrix: np.ndarray, edge_ratio: float, topk_per_node: int = 4) -> np.ndarray:
    num_nodes = fc_matrix.shape[0]
    if topk_per_node is not None:
        k = max(1, min(int(topk_per_node), num_nodes - 1))
        abs_fc = np.abs(fc_matrix)
        np.fill_diagonal(abs_fc, -np.inf)
        topk_idx = np.argpartition(abs_fc, -k, axis=1)[:, -k:]
        mask = np.zeros_like(fc_matrix, dtype=bool)
        row_idx = np.arange(num_nodes)[:, None]
        mask[row_idx, topk_idx] = True
        sparse_fc = np.where(mask, fc_matrix, 0.0).astype(np.float32)
        np.fill_diagonal(sparse_fc, 0.0)
        return sparse_fc

    if edge_ratio >= 1.0:
        return fc_matrix.astype(np.float32)

    num_edges = num_nodes * (num_nodes - 1)
    keep_edges = max(1, int(num_edges * float(edge_ratio)))
    flat = np.abs(fc_matrix).reshape(-1)
    topk_idx = np.argpartition(flat, -keep_edges)[-keep_edges:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[topk_idx] = True
    sparse_fc = np.where(mask.reshape(fc_matrix.shape), fc_matrix, 0.0).astype(np.float32)
    np.fill_diagonal(sparse_fc, 0.0)
    return sparse_fc


def _zscore_node_features(node_feats: np.ndarray) -> np.ndarray:
    mean = node_feats.mean(axis=1, keepdims=True)
    std = node_feats.std(axis=1, keepdims=True)
    std[std < EPS] = 1.0
    return ((node_feats - mean) / std).astype(np.float32)


def _select_node_features(node_feats: np.ndarray, fc_matrix: np.ndarray, node_feat_transform: str) -> np.ndarray:
    if node_feat_transform == 'timeseries':
        return _zscore_node_features(node_feats)
    if node_feat_transform == 'raw_timeseries':
        return node_feats.astype(np.float32)
    if node_feat_transform == 'pearson':
        return fc_matrix.astype(np.float32)
    raise NotImplementedError(f'Unsupported node_feat_transform: {node_feat_transform}')


def construct_dataset(dataName, edge_ratio, node_feat_transform='timeseries', topk_per_node: int = 4):
    """
    预期 meta_info:
      - name: 数据集名（用于输出）
      - edge_ratio: (0,1] 保留 |权重| 最大的比例
      - node_feat_transform: 支持 'timeseries'、'raw_timeseries'、'pearson'
    """
    data_name = dataName
    feat_dir  = os.path.join(BASEDIR,"dataset", data_name)

    # 1) 找到两类文件：时间序列 & 相关矩阵
    ts_paths = sorted(glob.glob(os.path.join(feat_dir, '*', '*_schaefer100_features_timeseries.mat'), recursive=True))
    fc_paths = sorted(glob.glob(os.path.join(feat_dir, '*', '*_schaefer100_correlation_matrix.mat'), recursive=True))

    ts_map = { _key_from_path(p): p for p in ts_paths }
    fc_map = { _key_from_path(p): p for p in fc_paths }

    keys = sorted(set(ts_map) & set(fc_map))
    if not keys:
        raise RuntimeError("未找到可配对的时间序列与相关矩阵文件（检查文件命名与路径）")

    print(f"发现 {len(keys)} 个可配对样本（TS & FC）")

    G_dataset = []
    Labels = []
    group2idx = {}
    # 2) 逐样本：TS -> 节点特征(N×T)，FC -> 边特征（非零建边）
    for k in tqdm(keys):
        ts_path = ts_map[k]
        fc_path = fc_map[k]

        try:
            ts = _load_mat(ts_path, key="data")
            node_feats = _ensure_node_by_time(ts)
        except Exception as exc:
            print(f"[WARN] 读取时间序列失败，跳过 {ts_path}: {exc}")
            continue

        N, T = node_feats.shape

        try:
            fc_matrix = _load_fc_matrix(fc_path, num_nodes=N)
        except Exception as exc:
            print(f"[WARN] 读取相关矩阵失败，跳过 {fc_path}: {exc}")
            continue

        sparse_fc_matrix = _sparsify_fc_matrix(fc_matrix, edge_ratio=edge_ratio, topk_per_node=topk_per_node)

        # 非零建边
        G = nx.from_numpy_array(sparse_fc_matrix, create_using=nx.DiGraph)
        g = dgl.from_networkx(G, edge_attrs=['weight'])

        g.ndata['N_features'] = torch.from_numpy(node_feats.astype(np.float32))
        g.ndata['FC_features'] = torch.from_numpy(fc_matrix.astype(np.float32))
        g.edata['E_features'] = g.edata.pop('weight').float()

        subject_name = os.path.basename(ts_path).replace('_schaefer100_features_timeseries.mat', '')
        metadata_row = get_subject_row(data_name, subject_name)
        if metadata_row is not None:
            Labels.append(encode_binary_label(metadata_row['Group']))
        else:
            # Fallback for datasets without Metadata-V5 support.
            m = re.findall(r'sub-([A-Za-z]+)', os.path.basename(ts_path))
            group = m[0] if m else 'default'
            if group not in group2idx:
                group2idx[group] = len(group2idx)
            Labels.append(group2idx[group])

        G_dataset.append(g)

    print('初始样本数:', len(G_dataset))

    # 1) 找到含“全零节点”的图，并统计最小时间长度
    error_case = []
    min_feat_dim = G_dataset[0].ndata['N_features'].shape[-1] if G_dataset else 0
    max_feat_dim = G_dataset[0].ndata['N_features'].shape[-1] if G_dataset else 0
    for i in range(len(G_dataset)):
        nf = G_dataset[i].ndata['N_features']                  # (N,T)
        has_all_zero_nodes = (~(nf != 0).any(dim=-1)).any().item()
        if has_all_zero_nodes:
            error_case.append(i)
        if nf.shape[-1] < min_feat_dim:
            min_feat_dim = nf.shape[-1]
        if nf.shape[-1] > max_feat_dim:
            max_feat_dim = nf.shape[-1]
    print("error_case:", error_case, " | min_feat_dim:", min_feat_dim , "| max_feat_dim:",max_feat_dim)

    # 2) 过滤掉异常图 & 同步 Labels &  
    keep = [i for i in range(len(G_dataset)) if i not in set(error_case)]
    G_dataset = [G_dataset[i] for i in keep]
    Labels = [Labels[i] for i in keep]

    for i in range(len(G_dataset)):
        nf = G_dataset[i].ndata['N_features']
        if nf.shape[-1] > min_feat_dim:
            start = (nf.shape[-1] - min_feat_dim) // 2
            nf = nf[:, start:start + min_feat_dim]
        elif nf.shape[-1] < min_feat_dim:
            pad = min_feat_dim - nf.shape[-1]
            nf = torch.cat([nf, nf[:, -1:].repeat(1, pad)], dim=-1)
        G_dataset[i].ndata['N_features'] = nf

    # 3) 写入运行时使用的特征视图
    edge_counts = []
    for i in tqdm(range(len(G_dataset))):
        G_dataset[i].edata['feat'] = G_dataset[i].edata['E_features'].unsqueeze(-1).clone()
        selected_feat = _select_node_features(
            G_dataset[i].ndata['N_features'].cpu().numpy(),
            G_dataset[i].ndata['FC_features'].cpu().numpy(),
            node_feat_transform=node_feat_transform
        )
        G_dataset[i].ndata['feat'] = torch.from_numpy(selected_feat).clone()
        edge_counts.append(int(G_dataset[i].num_edges()))

    if edge_counts:
        print(
            f'edge_count stats -> min: {min(edge_counts)}, '
            f'avg: {sum(edge_counts) / len(edge_counts):.2f}, max: {max(edge_counts)}'
        )

  # ---------------- 保存 bin ---------------- #
    out_dir = os.path.join(BASEDIR, 'bin_dataset')
    os.makedirs(out_dir, exist_ok=True)
    Labels = torch.LongTensor(Labels)
    save_graphs(BASEDIR+ "/bin_dataset/" + data_name + ".bin", G_dataset, {"glabel": Labels})
    del Labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct brain graph bins from schaefer100 files.')
    parser.add_argument('--datasets', nargs='+', default=['abide'], help='Dataset names under GOOD/data/dataset.')
    parser.add_argument('--edge-ratio', type=float, default=0.2)
    parser.add_argument('--node-feat-transform', default='timeseries')
    parser.add_argument('--topk-per-node', type=int, default=4)
    args = parser.parse_args()

    for data_name in args.datasets:
        construct_dataset(
            data_name,
            args.edge_ratio,
            node_feat_transform=args.node_feat_transform,
            topk_per_node=args.topk_per_node,
        )
    print('Done!')
