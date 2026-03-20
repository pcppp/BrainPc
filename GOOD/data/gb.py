def apply_gb_coarsening_to_dgl(g_dgl, label, ball_r=0.5):
    """
    对单个 DGL 图应用粒球粗化（修复版）
    
    Args:
        g_dgl: DGL图对象，包含 ndata['feat'] 和 edata['E_features']
        label: 图标签 (int)
        ball_r: 粗化比例
    
    Returns:
        粗化后的 DGL 图 (粒球作为节点)
    """
    from torch_geometric.data import Data
    import torch
    import dgl
    import numpy as np
    
    # 1️⃣ DGL 转 PyG (gb_division 需要)
    x = g_dgl.ndata['feat'].clone()  # (N, feat_dim)
    edge_index = torch.stack(g_dgl.edges(), dim=0)  # (2, E)
    
    # 构造 PyG Data
    y_tensor = torch.zeros(x.shape[0], dtype=torch.long)
    y_tensor.fill_(label)  # 所有节点标签一致
    
    pyg_data = Data(x=x, edge_index=edge_index, y=y_tensor)
    pyg_data.test_mask = torch.zeros(x.shape[0], dtype=torch.bool)  # gb_division 需要
    
    # 2️⃣ 调用粒球划分
    try:
        from gb_division import gb_division
        args = GBArgs(ball_r=ball_r, noisy=0)
        gb_result, gb_list, _ = gb_division(pyg_data, args)
    except Exception as e:
        print(f"⚠️ 粒球划分失败，跳过粗化: {e}")
        import traceback
        traceback.print_exc()
        return g_dgl  # 返回原图
    
    # 3️⃣ 提取粒球特征和边
    gb_features_avg = torch.tensor(gb_result['gb_features'], dtype=torch.float32)  # (num_balls, old_feat_dim)
    gb_adj = torch.tensor(gb_result['adj'], dtype=torch.long)  # (2, num_ball_edges)
    
    num_balls = gb_features_avg.shape[0]
    
    # 🔧 关键修复：重新计算粒球之间的Pearson相关性
    # gb_features_avg 是 (num_balls, old_feat_dim)，比如 (20, 100)
    # 我们需要计算粒球之间的相关性，得到 (num_balls, num_balls)，比如 (20, 20)
    
    if num_balls > 1:
        # 计算粒球之间的Pearson相关系数
        gb_features_np = gb_features_avg.cpu().numpy()  # (num_balls, old_feat_dim)
        gb_corr = np.corrcoef(gb_features_np, rowvar=True).astype(np.float32)  # (num_balls, num_balls)
        gb_features = torch.from_numpy(gb_corr)
        print(f"  ✓ 粒球特征: {gb_features_avg.shape} → {gb_features.shape} (重新计算Pearson相关性)")
    else:
        # 只有1个粒球时，特征就是它自己与自己的相关性(=1)
        gb_features = torch.ones((1, 1), dtype=torch.float32)
        print(f"  ⚠️ 只有1个粒球，特征设为 [1]")
    
    # 4️⃣ 创建新的 DGL 图 (粒球图)
    if gb_adj.shape[1] > 0:
        g_coarsened = dgl.graph((gb_adj[0], gb_adj[1]), num_nodes=num_balls)
    else:
        g_coarsened = dgl.graph(([], []), num_nodes=num_balls)
    
    # 5️⃣ 设置粒球特征 (保持字段名一致)
    g_coarsened.ndata['feat'] = gb_features  # (num_balls, num_balls)
    g_coarsened.ndata['N_features'] = gb_features
    
    # 6️⃣ 设置边特征 (简单设为1，因为gb_division不处理边特征)
    if g_coarsened.num_edges() > 0:
        g_coarsened.edata['E_features'] = torch.ones(g_coarsened.num_edges(), dtype=torch.float32)
        g_coarsened.edata['feat'] = g_coarsened.edata['E_features'].unsqueeze(-1)
    
    # 打印粗化信息
    if g_dgl.num_nodes() != num_balls:
        print(f"  ✓ {g_dgl.num_nodes()}节点 → {num_balls}粒球 (边: {g_dgl.num_edges()}→{g_coarsened.num_edges()})")
    
    return g_coarsened
