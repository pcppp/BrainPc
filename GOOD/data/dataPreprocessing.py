import os, re, glob, json, csv
import numpy as np
import scipy.io
import torch
import dgl
import networkx as nx
from tqdm import tqdm
from dgl.data.utils import save_graphs
import toponetx as tnx
BASEDIR = 'GOOD/data'
# ---------------------------------------------------------------------------------------------
def _key_from_path(p: str) -> str:
    """把文件名标准化成 subject key，用于 TS/FC 配对"""
    b = os.path.basename(p)
    b = re.sub(r'_schaefer100_(features_timeseries|correlation_matrix)\.mat$', '', b)
    return b

def _load_mat(path: str, key: str = "data") -> np.ndarray:
    m = scipy.io.loadmat(path)
    x = np.asarray(m[key], dtype=np.float32)
    return x

def construct_dataset(dataName, edge_ratio, node_feat_transform='pearson'):
    """
    预期 meta_info:
      - name: 数据集名（用于输出）
      - edge_ratio: (0,1] 保留 |权重| 最大的比例
      - node_feat_transform: 目前支持 'pearson'
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
    FC_list = []
    # 2) 逐样本：TS -> 节点特征(N×T)，FC -> 边特征（非零建边）
    for k in tqdm(keys):
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

        # 读入现成相关矩阵 FC (N×N) 作为边权
        FC = np.corrcoef(node_feats)
        # FC = _load_mat(fc_path, key="data")        # 期望 (N,N)
        FC_list.append(FC)
        if FC.shape != (N, N):
            print(f"[WARN] 尺寸不匹配，跳过：TS(N={N}) vs FC{FC.shape} in {k}")
            continue
        np.fill_diagonal(FC, 0.0)

        # 非零建边
        G = nx.from_numpy_array(FC, create_using=nx.DiGraph)
        g = dgl.from_networkx(G, edge_attrs=['weight'])

        # 写入特征
        g.ndata['N_features'] = torch.from_numpy(node_feats)             # (N,T)
        g.edata['E_features'] = g.edata.pop('weight').float()            # (E,)

        # 简单 label：按文件名里 'sub-XXXX' 提取站点/分组（按你原来的逻辑）
        name = os.path.basename(ts_path)
        m = re.findall(r'sub-([A-Za-z]+)', name)
        group = m[0] if m else 'default'
        if group not in group2idx.keys():
            group2idx[group] = len(group2idx.keys())
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
    # 时间维裁齐（裁到最短 T）
    # for i in range(len(G_dataset)):
    #     G_dataset[i].ndata['N_features'] = G_dataset[i].ndata['N_features'][:, :min_feat_dim]
    # 时间维Padding（扩充最大 T）
    # for i in range(len(G_dataset)):
    #     nf = G_dataset[i].ndata['N_features']
    #     if nf.shape[-1] < max_feat_dim:
    #         padding = max_feat_dim - nf.shape[-1]
    #         nf = torch.cat([nf, torch.zeros((nf.shape[0], padding))], dim=-1)
    #     G_dataset[i].ndata['N_features'] = nf
    # 3) 稀疏化：仅删除弱边（按 |w| ），不改权重
    edge_ratio = float(edge_ratio)
    import copy
    G_origin_dataset = copy.deepcopy(G_dataset)
    for i in tqdm(range(len(G_dataset))):
        e = G_dataset[i].edata['E_features'].float()  # (E,)

        if edge_ratio < 1.0 and e.numel() > 0:
            M = e.numel()
            k_keep = max(1, int(M * edge_ratio))      # 保留的边数

            # 取 |w| 最大的 k_keep 条边的索引（完全在 GPU 上完成）
            _, keep_idx = torch.topk(e.abs(), k_keep, largest=True, sorted=False)  # (k_keep,)

            keep_mask = torch.zeros(M, dtype=torch.bool, device=e.device)
            keep_mask[keep_idx] = True
            drop_idx = (~keep_mask).nonzero(as_tuple=False).squeeze(1)            # (M - k_keep,)

            # 原地删除弱边
            G_dataset[i].remove_edges(drop_idx)
    for i in tqdm(range(len(G_dataset))):
        e = G_dataset[i].edata['E_features'].float()  # (E,)

        if edge_ratio < 1.0 and e.numel() > 0:
            M = e.numel()
            k_keep = max(1, int(M * edge_ratio))      # 保留的边数

            # 取 |w| 最大的 k_keep 条边的索引（完全在 GPU 上完成）
            _, keep_idx = torch.topk(e.abs(), k_keep, largest=True, sorted=False)  # (k_keep,)

            keep_mask = torch.zeros(M, dtype=torch.bool, device=e.device)
            keep_mask[keep_idx] = True
            drop_idx = (~keep_mask).nonzero(as_tuple=False).squeeze(1)            # (M - k_keep,)

            # 原地删除弱边
            G_dataset[i].remove_edges(drop_idx)

        # 下游如需 edata['feat']：做一个列向量别名（同步在删边之后）
        G_dataset[i].edata['feat'] = G_dataset[i].edata['E_features'].unsqueeze(-1).clone()

        # 5) 节点特征变换
        if node_feat_transform == 'pearson':
            nf_np = G_dataset[i].ndata['N_features'].cpu().numpy()   # (N,T)
            nf_corr = np.corrcoef(nf_np, rowvar=True).astype(np.float32)  # (N,N)
            # G_dataset[i].ndata['feat'] = G_dataset[i].ndata['N_features']
            G_dataset[i].ndata['feat'] = torch.from_numpy(nf_corr).clone()
            G_origin_dataset[i].ndata['feat'] = torch.from_numpy(nf_corr).clone()
            G_dataset[i].ndata['N_features'] = torch.from_numpy(nf_corr).clone()
            G_origin_dataset[i].ndata['N_features'] = torch.from_numpy(nf_corr).clone()
        else:
            raise NotImplementedError
  # ---------------- 保存 bin ---------------- #
    out_dir = os.path.join(BASEDIR, 'bin_dataset')
    os.makedirs(out_dir, exist_ok=True)
    Labels = torch.LongTensor(Labels)          # 你的标签
    save_graphs(BASEDIR+ "/bin_dataset/" + data_name + ".bin", G_dataset, {"glabel": Labels})
    del Labels
if __name__ == '__main__':
    error_name = []
    file_name_list = ['abide']

    for data_name in file_name_list:
        construct_dataset(data_name, 0.2)
        # except:
        #     print('[ERROR]: ' + data_name)
        #     error_name.append(data_name)
    if not len(error_name):
        print('Done!')
