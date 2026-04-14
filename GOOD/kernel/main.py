r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import itertools
import os
from pathlib import Path
import time
from typing import Tuple, Union
import numpy as np
import pandas as pd
import torch.nn
from torch.utils.data import DataLoader
import copy

from tqdm import tqdm
from GOOD import config_summoner
from GOOD.data import load_dataset, create_dataloader
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.networks.model_manager import load_model
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import CommonArgs, Munch, process_configs
from GOOD.utils.initial import reset_random_seed
# from GOOD.utils.logger import load_logger
from GOOD.definitions import OOM_CODE
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from GOOD.utils.visualize_tsne import extract_embeddings, plot_tsne_comparison

def initialize_model_dataset(config: Union[CommonArgs, Munch], fold: int = 0) -> Tuple[torch.nn.Module, Union[dict, DataLoader]]:
    r"""
    Fix random seeds and initialize a GNN and a dataset. (For project use only)

    Returns:
        A GNN and a data loader.
    """
    try:
        # Initial
        reset_random_seed(config)

        print(f'#IN#\n-----------------------------------\n    Task: {config.task}\n'
              f'{time.asctime(time.localtime(time.time()))}')
        # Load dataset
        print(f'#IN#Load Dataset {config.dataset.dataset_name}')

        dataset = load_dataset(config.dataset.dataset_name, config, fold)
        loader = create_dataloader(dataset, config)

        # Load model
        print(f'#IN#Loading model...')
        model = load_model(config.model.model_name, config)

        return model, loader
    except Exception as e:
        print(f'#ERROR# 初始化模型或数据集时发生错误: {str(e)}')
        import traceback
        traceback.print_exc()
        raise  # 重新抛出异常


def compute_10fold_metrics(ckpts):
    train_scores = []
    id_val_scores = []
    id_test_scores = []
    ood_val_scores = []
    ood_test_scores = []
    val_scores = []
    test_scores = []
    for ckpt in ckpts:
        train_scores.append(ckpt['train_score'])
        id_val_scores.append(ckpt['id_val_score'])
        id_test_scores.append(ckpt['id_test_score'])
        ood_val_scores.append(ckpt['ood_val_score'])
        ood_test_scores.append(ckpt['ood_test_score'])
        val_scores.append(ckpt.get('mixed_val_score', ckpt['val_score']))
        test_scores.append(ckpt.get('mixed_test_score', ckpt['test_score']))

    # compute the mean and std of these metrics, keep four decimal places
    train_mean = torch.mean(torch.tensor(train_scores))
    train_std = torch.std(torch.tensor(train_scores))
    id_val_mean = torch.mean(torch.tensor(id_val_scores))
    id_val_std = torch.std(torch.tensor(id_val_scores))
    id_test_mean = torch.mean(torch.tensor(id_test_scores))
    id_test_std = torch.std(torch.tensor(id_test_scores))
    ood_val_mean = torch.mean(torch.tensor(ood_val_scores))
    ood_val_std = torch.std(torch.tensor(ood_val_scores))
    ood_test_mean = torch.mean(torch.tensor(ood_test_scores))
    ood_test_std = torch.std(torch.tensor(ood_test_scores))
    val_score_mean = torch.mean(torch.tensor(val_scores))
    val_score_std = torch.std(torch.tensor(val_scores))
    test_score_mean = torch.mean(torch.tensor(test_scores))
    test_score_std = torch.std(torch.tensor(test_scores))
    results = {
        'train_mean': round(train_mean.item(), 4),
        'train_std': round(train_std.item(), 4),
        'id_val_mean': round(id_val_mean.item(), 4),
        'id_val_std': round(id_val_std.item(), 4),
        'id_test_mean': round(id_test_mean.item(), 4),
        'id_test_std': round(id_test_std.item(), 4),
        'ood_val_mean': round(ood_val_mean.item(), 4),
        'ood_val_std': round(ood_val_std.item(), 4),
        'ood_test_mean': round(ood_test_mean.item(), 4),
        'ood_test_std': round(ood_test_std.item(), 4),
        'val_score_mean': round(val_score_mean.item(), 4),
        'val_score_std': round(val_score_std.item(), 4),
        'test_score_mean': round(test_score_mean.item(), 4),
        'test_score_std': round(test_score_std.item(), 4)
    }
    return results


def compute_mixed_10fold_metrics(ckpts, id_ckpts):
    val_scores = []
    test_scores = []
    test_precision = []
    test_recall = []
    test_f1 = []
    test_roc_auc = []
    for ckpt, id_ckpt in zip(ckpts, id_ckpts):
        val_scores.append((id_ckpt['id_val_score'] * id_ckpt['id_val_subject_num'] + ckpt['ood_val_score'] * ckpt['ood_val_subject_num']) / (id_ckpt['id_val_subject_num'] + ckpt['ood_val_subject_num']))
        test_scores.append((id_ckpt['id_test_score'] * id_ckpt['id_test_subject_num'] + ckpt['ood_test_score'] * ckpt['ood_test_subject_num']) / (id_ckpt['id_test_subject_num'] + ckpt['ood_test_subject_num']))
        test_precision.append((id_ckpt['id_test_precision'] * id_ckpt['id_test_subject_num'] + ckpt['ood_test_precision'] * ckpt['ood_test_subject_num']) / (id_ckpt['id_test_subject_num'] + ckpt['ood_test_subject_num']))
        test_recall.append((id_ckpt['id_test_recall'] * id_ckpt['id_test_subject_num'] + ckpt['ood_test_recall'] * ckpt['ood_test_subject_num']) / (id_ckpt['id_test_subject_num'] + ckpt['ood_test_subject_num']))
        test_f1.append((id_ckpt['id_test_f1'] * id_ckpt['id_test_subject_num'] + ckpt['ood_test_f1'] * ckpt['ood_test_subject_num']) / (id_ckpt['id_test_subject_num'] + ckpt['ood_test_subject_num']))
        test_roc_auc.append((id_ckpt['id_test_roc_auc'] * id_ckpt['id_test_subject_num'] + ckpt['ood_test_roc_auc'] * ckpt['ood_test_subject_num']) / (id_ckpt['id_test_subject_num'] + ckpt['ood_test_subject_num']))

    # compute the mean and std of these metrics, keep four decimal places
    val_score_mean = torch.mean(torch.tensor(val_scores))
    val_score_std = torch.std(torch.tensor(val_scores))
    test_score_mean = torch.mean(torch.tensor(test_scores))
    test_score_std = torch.std(torch.tensor(test_scores))
    test_precision_mean = torch.mean(torch.tensor(test_precision))
    test_precision_std = torch.std(torch.tensor(test_precision))
    test_recall_mean = torch.mean(torch.tensor(test_recall))
    test_recall_std = torch.std(torch.tensor(test_recall))
    test_f1_mean = torch.mean(torch.tensor(test_f1))
    test_f1_std = torch.std(torch.tensor(test_f1))
    test_roc_auc_mean = torch.mean(torch.tensor(test_roc_auc))
    test_roc_auc_std = torch.std(torch.tensor(test_roc_auc))
    results = {
        'val_score_mean': round(val_score_mean.item(), 4),
        'val_score_std': round(val_score_std.item(), 4),
        'test_score_mean': round(test_score_mean.item(), 4),
        'test_score_std': round(test_score_std.item(), 4),
        'test_precision_mean': round(test_precision_mean.item(), 4),
        'test_precision_std': round(test_precision_std.item(), 4),
        'test_recall_mean': round(test_recall_mean.item(), 4),
        'test_recall_std': round(test_recall_std.item(), 4),
        'test_f1_mean': round(test_f1_mean.item(), 4),
        'test_f1_std': round(test_f1_std.item(), 4),
        'test_roc_auc_mean': round(test_roc_auc_mean.item(), 4),
        'test_roc_auc_std': round(test_roc_auc_std.item(), 4)
    }
    return results

def run_10fold_once(config):
    id_ckpts, ckpts = [], []
    for i in range(10):
        print(f'\nFold {i + 1}')
        process_configs(config, i)

        config.task = 'train'
        model, loader = initialize_model_dataset(config, i)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
        pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
        if i == 0:
            view_model_param(pipeline.model)
        # ================= [可视化核心代码 Start] =================
        # 仅在第 1 折 (Fold 0) 进行可视化，节省时间
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        emb_before, y_before = None, None
        if i == 0:
            print(">>> [t-SNE] Extracting embeddings BEFORE training...")
            # 注意：loader 通常是字典，我们需要选一个用于可视化的子集
            # 优先使用 'id_val' 或 'val'，如果没有则用 'train'
            vis_loader_key = 'id_val' if 'id_val' in loader else ('val' if 'val' in loader else 'train')
            vis_loader = loader[vis_loader_key]
            
            try:
                # 传入 config 以便内部获取 device 或其他参数
                emb_before, y_before = extract_embeddings(model, vis_loader, config.device)
            except Exception as e:
                print(f"#WARNING# t-SNE extraction failed before training: {e}")
        # ================= [可视化核心代码 End] =================
        # 训练
        pipeline.load_task(fold=i)

        # ================= [可视化: 训练前后对比] =================
        if i == 0 and emb_before is not None:
            print(">>> [t-SNE] Extracting embeddings AFTER training...")
            try:
                emb_after, y_after = extract_embeddings(model, vis_loader, config.device, max_samples=500)
                if emb_after is not None:
                    plot_name = f"tsne_fold{i}_before_vs_after.png"
                    save_path = log_dir / plot_name
                    plot_tsne_comparison(
                        emb_before, y_before,
                        emb_after, y_after,
                        save_path=save_path
                    )
                    print(f">>> [Success] t-SNE对比图已保存: {plot_name}")
            except Exception as e:
                print(f"#WARNING# t-SNE failed after training: {e}")
        # ================= [可视化 End] =================
        # 测试（按你原逻辑）
        if config.task == 'train':
            pipeline.task = 'test'
            id_ckpt, ckpt = pipeline.load_task(fold=i)
            id_ckpts.append(id_ckpt)
            ckpts.append(ckpt)

    id_ckpt_results = compute_10fold_metrics(id_ckpts)
    ckpt_results = compute_10fold_metrics(ckpts)
    mixed_results = compute_mixed_10fold_metrics(ckpts, id_ckpts)

    # 保留你原来的打印（可选）
    print('#IN#\n\nID-ckpt results:')
    print('#IN#Train: {} ± {}'.format(id_ckpt_results['train_mean'], id_ckpt_results['train_std']))
    print('#IN#ID-val: {} ± {}'.format(id_ckpt_results['id_val_mean'], id_ckpt_results['id_val_std']))
    print('#IN#ID-test: {} ± {}'.format(id_ckpt_results['id_test_mean'], id_ckpt_results['id_test_std']))
    print('#IN#OOD-val: {} ± {}'.format(id_ckpt_results['ood_val_mean'], id_ckpt_results['ood_val_std']))
    print('#IN#OOD-test: {} ± {}'.format(id_ckpt_results['ood_test_mean'], id_ckpt_results['ood_test_std']))
    print('#IN#Val: {} ± {}'.format(id_ckpt_results['val_score_mean'], id_ckpt_results['val_score_std']))
    print('#IN#Test: {} ± {}'.format(id_ckpt_results['test_score_mean'], id_ckpt_results['test_score_std']))

    print('#IN#\nOOD-ckpt results:')
    print('#IN#Train: {} ± {}'.format(ckpt_results['train_mean'], ckpt_results['train_std']))
    print('#IN#ID-val: {} ± {}'.format(ckpt_results['id_val_mean'], ckpt_results['id_val_std']))
    print('#IN#ID-test: {} ± {}'.format(ckpt_results['id_test_mean'], ckpt_results['id_test_std']))
    print('#IN#OOD-val: {} ± {}'.format(ckpt_results['ood_val_mean'], ckpt_results['ood_val_std']))
    print('#IN#OOD-test: {} ± {}'.format(ckpt_results['ood_test_mean'], ckpt_results['ood_test_std']))
    print('#IN#Val: {} ± {}'.format(ckpt_results['val_score_mean'], ckpt_results['val_score_std']))
    print('#IN#Test: {} ± {}'.format(ckpt_results['test_score_mean'], ckpt_results['test_score_std']))

    print('#IN#\nMixed results:')
    print('#IN#Val: {} ± {}'.format(mixed_results['val_score_mean'], mixed_results['val_score_std']))
    print('#IN#Test: {} ± {}'.format(mixed_results['test_score_mean'], mixed_results['test_score_std']))
    print('#IN#Test precision: {} ± {}'.format(mixed_results['test_precision_mean'], mixed_results['test_precision_std']))
    print('#IN#Test recall: {} ± {}'.format(mixed_results['test_recall_mean'], mixed_results['test_recall_std']))
    print('#IN#Test F1: {} ± {}'.format(mixed_results['test_f1_mean'], mixed_results['test_f1_std']))
    print('#IN#Test ROC AUC: {} ± {}'.format(mixed_results['test_roc_auc_mean'], mixed_results['test_roc_auc_std']))
    return id_ckpt_results, ckpt_results, mixed_results
def main():
    args = args_parser()
    base_config = config_summoner(args)

    # 网格
    lambda1_list = [0.1]   # entropy_trade_off [0.1, 0.01, 0.001]   
    lambda2_list = [1.0]     # trade_off [1.0, 0.1, 0.01]
    lambda3_list = [1.0]      # diffusion_trade_off [1.0, 0.5, 0.1] 
    epoch_list = [60]
    # lr_list = [2e-3,4e-3,6e-3,8e-3,1e-2]
    lr_list = [1e-3]
    total = len(lambda1_list) * len(lambda2_list) * len(lambda3_list)*len(epoch_list)*len(lr_list)
   
    # XML
    log_dir = Path("logs"); 
    log_dir.mkdir(exist_ok=True)
    xml_path = log_dir / "grid_results.xml"
     # Excel 文件路径
    excel_path =  log_dir /"grid_results.xlsx"
     # 创建 DataFrame 来存储结果
    columns = ["Parameter Combination", "Test", "Test_precision", "Test_recall", "Test_F1", "Test_ROC_AUC"]
    results_df = pd.DataFrame(columns=columns)
    row_data_list = []
    for (l1, l2, l3,epoch,lr) in tqdm(itertools.product(lambda1_list, lambda2_list, lambda3_list,epoch_list,lr_list), total=total, desc="Grid"):
        # 拷贝 config，写入三个超参
        config = copy.deepcopy(base_config)
        # 注意：这里假设 config.ood 下已有这些字段（Munch/Namespace）
        config.ood.entropy_trade_off   = float(l1)  # λ1
        config.ood.trade_off           = float(l2)  # λ2
        config.ood.diffusion_trade_off = float(l3)  # λ3
        config.train.max_epoch = int(epoch)
        config.train.lr = float(lr)
        print(f"\n=== Run with λ1={l1}, λ2={l2}, λ3={l3} epoch_list ={epoch} lr_list = {lr} === ")
        id_res, ood_res, mix_res = run_10fold_once(config)

        row_data = {
            "Parameter Combination": f"λ1={l1}, λ2={l2}, λ3={l3},epoch={epoch},lr={lr}",
            "Test": f"{mix_res['test_score_mean']} ± {mix_res['test_score_std']}",
            "Test_precision": f"{mix_res['test_precision_mean']} ± {mix_res['test_precision_std']}",
            "Test_recall": f"{mix_res['test_recall_mean']} ± {mix_res['test_recall_std']}",
            "Test_F1": f"{mix_res['test_f1_mean']} ± {mix_res['test_f1_std']}",
            "Test_ROC_AUC": f"{mix_res['test_roc_auc_mean']} ± {mix_res['test_roc_auc_std']}",
        }
        # 将结果添加到 DataFrame
        row_data_list.append(row_data)

         # 保存结果到 Excel 文件
    results_df = pd.DataFrame(row_data_list)
    results_df.to_excel(excel_path, index=False)

    print(f"\nAll done. XML saved to: {xml_path}")
def goodtg():
    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'#E#{e}')
            exit(OOM_CODE)
        else:
            raise e

def view_model_param(model):
    # model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    # print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('Total parameters:', total_param)


if __name__ == '__main__':
    main()
