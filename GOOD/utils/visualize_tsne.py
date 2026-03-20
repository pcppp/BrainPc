"""
t-SNE可视化脚本：对比未训练和训练后的模型
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse


def extract_embeddings(model, loader, device, max_samples=1000):
    """
    专门适配 GNN OOD 模型的特征提取
    """
    model = model.to(device)
    model.eval()
    embeddings_list = []
    labels_list = []
    count = 0
    
    with torch.no_grad():
        for data in loader:
            if count >= max_samples:
                break
            
            data = data.to(device)
            
            # --- 关键修改：处理模型输出 ---
            try:
                # 很多对比学习模型在 forward 时需要 input_preprocess 后的数据
                # 如果你的 model 直接接受 data:
                output = model(data=data)

                # 如果输出是 tuple (z1, z2)，我们只取其中一个，或者取 model.encoder 的输出
                if isinstance(output, (tuple, list)):
                    # 通常取第一个视图，或者你可能有专门的 inference 接口
                    feature = output[0] 
                else:
                    feature = output

                # 如果 feature 还是 tuple (例如 (logits, embedding))，继续拆
                if isinstance(feature, (tuple, list)):
                    feature = feature[-1] # 假设最后一个是 embedding
                    
            except Exception as e:
                # 如果直接 model(data) 报错（因为预训练模型可能期待两个视图），
                # 你可能需要手动调用 backbone/encoder
                # feature = model.backbone(data.x, data.edge_index) 
                print(f"[Warning] Extraction failed: {e}")
                continue
            # ---------------------------

            # 确保 feature 是 2D [Batch, Dim]
            if feature.dim() > 2:
                feature = feature.mean(dim=1) # 若是节点级 embedding，聚合为图级
                
            embeddings_list.append(feature.cpu().numpy())
            
            # 确保 labels 存在
            if hasattr(data, 'y') and data.y is not None:
                labels_list.append(data.y.cpu().numpy())
            else:
                # 如果没有标签，生成伪标签以免报错
                labels_list.append(np.zeros(feature.shape[0]))
            
            count += feature.shape[0]

    if not embeddings_list:
        return None, None

    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # 展平 labels
    if labels.ndim > 1: labels = labels.flatten()
    
    return embeddings[:max_samples], labels[:max_samples]   
def plot_tsne_comparison(emb_before, labels_before, emb_after, labels_after, save_path):
    """
    鲁棒性修复版：自动对齐 embedding 和 label 的长度，防止 63 vs 126 维度不匹配报错
    """
    
    # --- 核心修复：分别计算两组数据的最小长度 ---
    # 防止 embedding 只有 63 个，但 label 有 126 个的情况
    len_before = min(len(emb_before), len(labels_before))
    len_after = min(len(emb_after), len(labels_after))
    
    # 再次取两者的交集，确保左右两张图的点数一致（为了美观，也可以不一致）
    # 这里我们为了最大化利用数据，分别处理左右图，只限制绘图上限
    limit = 1000 # 最大绘图点数，防卡顿
    
    n_before = min(len_before, limit)
    n_after = min(len_after, limit)
    
    print(f"[t-SNE] 对齐数据维度 -> Before: {n_before} (Emb:{len(emb_before)}, Lbl:{len(labels_before)}) | After: {n_after}")

    # --- 截断数据 ---
    # 必须同时截断 embedding 和 labels，保证下标对齐
    emb_b = emb_before[:n_before]
    y_b = labels_before[:n_before]
    
    emb_a = emb_after[:n_after]
    y_a = labels_after[:n_after]

    try:
        # 动态 Perplexity: 样本少时不能太大
        perp_b = min(30, max(5, n_before // 4))
        perp_a = min(30, max(5, n_after // 4))

        # 计算 t-SNE
        # 移除 n_iter 参数以兼容旧版 sklearn
        tsne_b = TSNE(n_components=2, perplexity=perp_b, random_state=42)
        emb_2d_before = tsne_b.fit_transform(emb_b)
        
        tsne_a = TSNE(n_components=2, perplexity=perp_a, random_state=42)
        emb_2d_after = tsne_a.fit_transform(emb_a)
        
    except Exception as e:
        print(f"[Error] t-SNE 计算崩溃: {e}")
        return

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 统一颜色盘 (基于 Before 的标签)
    unique_labels = np.unique(y_b)
    # 动态生成颜色
    if len(unique_labels) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
    # 建立 label -> color 映射，保证左右两图同一类别颜色一致
    label_to_color = {lbl: colors[i % len(colors)] for i, lbl in enumerate(unique_labels)}

    def plot_scatter(ax, emb, labels, title):
        curr_labels = np.unique(labels)
        for lbl in curr_labels:
            mask = labels == lbl
            c = label_to_color.get(lbl, 'gray') # 如果有新类别，用灰色
            c = c.reshape(1,-1) if isinstance(c, np.ndarray) and c.ndim==1 else c
            
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=c, label=f'Class {int(lbl)}',
                       alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        # 图例去重逻辑交给 matplotlib，但如果类别太多建议隐藏
        if len(curr_labels) < 15:
            ax.legend(loc='best', markerscale=1.5, fontsize='small')

    plot_scatter(ax1, emb_2d_before, y_b, 'Before Training')
    plot_scatter(ax2, emb_2d_after, y_a, 'After Training')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[Success] t-SNE 图已保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='tsne_comparison.png')
    parser.add_argument('--max_samples', type=int, default=500)
    args = parser.parse_args()
    
    # 这里需要根据你的实际代码结构导入
    # 以下是示例代码框架
    print("Loading config and data...")
    # config = load_your_config(args.config)
    # model = create_your_model(config)
    # loader = create_your_dataloader(config)
    
    print("This is a template. Please integrate with your actual code.")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    print("t-SNE Visualization Script")
    print("="*50)
    # Uncomment when integrated with your code
    # main()
