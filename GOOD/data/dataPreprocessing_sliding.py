import os, re, glob, json, csv
import numpy as np
import scipy.io
import torch
import dgl
import networkx as nx
from tqdm import tqdm
from dgl.data.utils import save_graphs
import pywt

BASEDIR = 'GOOD/data'

def _key_from_path(p: str) -> str:
    """把文件名标准化成 subject key，用于 TS/FC 配对"""
    b = os.path.basename(p)
    b = re.sub(r'_schaefer100_(features_timeseries|correlation_matrix)\.mat$', '', b)
    return b

def _load_mat(path: str, key: str = "data") -> np.ndarray:
    m = scipy.io.loadmat(path)
    x = np.asarray(m[key], dtype=np.float32)
    return x

def construct_dataset(dataName,
                      edge_ratio=0.1,
                      node_feat_transform='timeseries',
                      use_wavelet=False):
    """
    预处理数据集：
      - node_feat_transform='timeseries': 返回Z-score标准化后的时间序列 (N, T)，让CNN自己学习时间模式
      - node_feat_transform='pearson': 返回Pearson相关矩阵 (N, N)  
      - edge_ratio: (0,1] 保留 |权重| 最大的比例
      - 边特征：Pearson相关系数已包含在E_features中
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
        raise RuntimeError("未找到可配对的时间序列与相关矩阵文件")

    print(f"发现 {len(keys)} 个可配对样本（TS & FC）")

    G_dataset = []
    Labels = []
    group2idx = {}
    
    # 2) 逐样本：TS -> 节点特征(N×T)，FC -> 边特征（Pearson相关系数）
    for k in tqdm(keys, desc="加载数据"):
        ts_path = ts_map[k]
        fc_path = fc_map[k]

        ts = _load_mat(ts_path, key="data")        # (T,N) 或 (N,T)
        if ts.ndim != 2:
            print(f"[WARN] {ts_path} 不是二维数据，跳过：{ts.shape}")
            continue

        # 统一成 节点×时间 (N×T)
        R, C = ts.shape
        if R >= C:          # 常见 (T,N)
            node_feats = ts.T          # -> (N,T)
        else:                # (N,T)
            node_feats = ts            # -> (N,T)
        N, T = node_feats.shape
        if use_wavelet:
            # 小波分解：提取近似系数 (Approximation Coefficients) 以获得平滑宏观信号
            # 使用 Daubechies 4 (db4) 小波
            feats_list = []
            for i in range(N):
                cA, _ = pywt.dwt(node_feats[i], 'db4')
                feats_list.append(cA)
            node_feats = np.array(feats_list, dtype=np.float32)
            N, T = node_feats.shape


        # 读入Pearson相关矩阵 FC (N×N) 作为边权
        # FC = _load_mat(fc_path, key="data")        # (N,N)
        # 改为直接从节点特征计算 Pearson 相关系数
        FC = np.corrcoef(node_feats)
        if FC.shape != (N, N):
            print(f"[WARN] 尺寸不匹配，跳过：TS(N={N}) vs FC{FC.shape} in {k}")
            continue
        np.fill_diagonal(FC, 0.0)

        # 构建图：FC作为边权（Pearson相关系数）
        G = nx.from_numpy_array(FC, create_using=nx.DiGraph)
        g = dgl.from_networkx(G, edge_attrs=['weight'])

        # 节点特征：原始时间序列 (N,T)
        g.ndata['N_features'] = torch.from_numpy(node_feats)
        # 边特征：Pearson相关系数
        g.edata['E_features'] = g.edata.pop('weight').float()

        # 提取标签（站点/分组）
        name = os.path.basename(ts_path)
        m = re.findall(r'sub-([A-Za-z]+)', name)
        group = m[0] if m else 'default'
        if group not in group2idx.keys():
            group2idx[group] = len(group2idx.keys())
        Labels.append(group2idx[group])

        G_dataset.append(g)

    print('初始样本数:', len(G_dataset))

    # 3) 过滤全零节点 & 统计时间维度
    error_case = []
    min_feat_dim = G_dataset[0].ndata['N_features'].shape[-1] if G_dataset else 0
    max_feat_dim = min_feat_dim
    
    for i in range(len(G_dataset)):
        nf = G_dataset[i].ndata['N_features']                  # (N,T)
        has_all_zero_nodes = (~(nf != 0).any(dim=-1)).any().item()
        if has_all_zero_nodes:
            error_case.append(i)
        T = nf.shape[-1]
        min_feat_dim = min(min_feat_dim, T)
        max_feat_dim = max(max_feat_dim, T)
        
    print(f"error_case: {error_case} | min_feat_dim: {min_feat_dim} | max_feat_dim: {max_feat_dim}")

    # 过滤异常样本
    keep = [i for i in range(len(G_dataset)) if i not in set(error_case)]
    G_dataset = [G_dataset[i] for i in keep]
    Labels = [Labels[i] for i in keep]

    # 4) 时间维裁齐到 min_feat_dim（所有样本统一长度，CNN才能处理）
    print(f"裁齐时间维度到 {min_feat_dim}...")
    for i in range(len(G_dataset)):
        nf = G_dataset[i].ndata['N_features']  # (N, T)
        T = nf.shape[-1]
        if T > min_feat_dim:
            # 中心裁剪，保留中间时间段
            start = (T - min_feat_dim) // 2
            nf = nf[:, start:start + min_feat_dim]
        elif T < min_feat_dim:
            # 理论上不会发生（因为裁到min），但以防万一
            padding = min_feat_dim - T
            nf = torch.cat([nf, nf[:, -1:].repeat(1, padding)], dim=-1)
        G_dataset[i].ndata['N_features'] = nf

    # 5) 边稀疏化：保留|权重|最大的边
    import copy
    G_origin_dataset = copy.deepcopy(G_dataset)
    edge_ratio = float(edge_ratio)
    
    for i in tqdm(range(len(G_dataset)), desc="边稀疏化"):
        e = G_dataset[i].edata['E_features'].float()  # (E,) Pearson相关系数

        if edge_ratio < 1.0 and e.numel() > 0:
            M = e.numel()
            k_keep = max(1, int(M * edge_ratio))

            _, keep_idx = torch.topk(e.abs(), k_keep, largest=True, sorted=False)
            keep_mask = torch.zeros(M, dtype=torch.bool, device=e.device)
            keep_mask[keep_idx] = True
            drop_idx = (~keep_mask).nonzero(as_tuple=False).squeeze(1)

            G_dataset[i].remove_edges(drop_idx)

        # 边特征别名
        G_dataset[i].edata['feat'] = G_dataset[i].edata['E_features'].unsqueeze(-1).clone()

    # 6) 节点特征变换
    for i in tqdm(range(len(G_dataset)), desc="节点特征变换"):
        nf_np = G_dataset[i].ndata['N_features'].cpu().numpy()   # (N,T)
        N, T = nf_np.shape
        
        if node_feat_transform == 'timeseries':
            # 返回Z-score标准化的时间序列 (N, T)
            # CNN将直接学习时间模式（卷积核=滑动窗口）
            mean = nf_np.mean(axis=1, keepdims=True)  # (N, 1)
            std = nf_np.std(axis=1, keepdims=True) + 1e-8
            nf_normalized = (nf_np - mean) / std  # Z-score归一化
            
            feat_tensor = torch.from_numpy(nf_normalized.astype(np.float32)).clone()
            G_dataset[i].ndata['feat'] = feat_tensor  # (N, T) 供CNN处理
            G_origin_dataset[i].ndata['feat'] = feat_tensor.clone()
            G_dataset[i].ndata['N_features'] = feat_tensor.clone()
            G_origin_dataset[i].ndata['N_features'] = feat_tensor.clone()
            
        elif node_feat_transform == 'pearson':
            # Pearson相关矩阵 (N, N)
            nf_corr = np.corrcoef(nf_np, rowvar=True).astype(np.float32)
            feat_tensor = torch.from_numpy(nf_corr).clone()
            G_dataset[i].ndata['feat'] = feat_tensor
            G_origin_dataset[i].ndata['feat'] = feat_tensor.clone()
            G_dataset[i].ndata['N_features'] = feat_tensor.clone()
            G_origin_dataset[i].ndata['N_features'] = feat_tensor.clone()
        else:
            raise NotImplementedError(f"Unsupported node_feat_transform: {node_feat_transform}")
    
    print(f'完成! 节点特征维度: {G_dataset[0].ndata["N_features"].shape} | 边特征维度: {G_dataset[0].edata["feat"].shape}')

    # 7) 保存数据集
    out_dir = os.path.join(BASEDIR, 'bin_time_dataset')
    os.makedirs(out_dir, exist_ok=True)
    Labels = torch.LongTensor(Labels)
    save_graphs(os.path.join(BASEDIR, "bin_time_dataset", f"{data_name}.bin"), G_dataset, {"glabel": Labels})
    print(f'数据已保存到: {os.path.join(BASEDIR, "bin_time_dataset", f"{data_name}.bin")}')

if __name__ == '__main__':
    file_name_list = ['abide']
    for data_name in file_name_list:
        # 使用 timeseries 模式：返回Z-score标准化的时间序列 (N, T)
        # CNN将直接对时间序列做卷积，自动学习时间模式
        # 边特征：Pearson相关系数（已在E_features中）
        construct_dataset(data_name, node_feat_transform='timeseries', use_wavelet=True)
    print('Done!')
