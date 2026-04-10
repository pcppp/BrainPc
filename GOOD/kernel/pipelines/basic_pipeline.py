r"""Training pipeline: training/evaluation structure, batch training.
"""
import datetime
import os
import shutil
from typing import Dict
from typing import Union

import numpy as np
import torch
import torch.nn
from munch import Munch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.args import CommonArgs
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.register import register
from GOOD.utils.train import nan2zero_get_mask


@register.pipeline_register
class Pipeline:
    r"""
    Kernel pipeline.

    Args:
        task (str): Current running task. 'train' or 'test'
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    """

    def __init__(self, task: str, model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]],
                 ood_algorithm: BaseOODAlg,
                 config: Union[CommonArgs, Munch]):
        super(Pipeline, self).__init__()
        self.task: str = task
        self.model: torch.nn.Module = model
        self.loader: Union[DataLoader, Dict[str, DataLoader]] = loader
        self.ood_algorithm: BaseOODAlg = ood_algorithm
        self.config: Union[CommonArgs, Munch] = config

    def _augment_contrastive_view(self, data: Batch, edge_drop_rate: float, feature_mask_rate: float, noise_std: float) -> Batch:
        graph_list = []
        for graph in data.to_data_list():
            aug = graph.clone()
            num_edges = aug.edge_index.size(1)
            if num_edges > 0 and edge_drop_rate > 0:
                keep_mask = torch.rand(num_edges, device=aug.edge_index.device) > edge_drop_rate
                if keep_mask.sum() == 0:
                    keep_mask[torch.randint(0, num_edges, (1,), device=aug.edge_index.device)] = True
                aug.edge_index = aug.edge_index[:, keep_mask]
                if getattr(aug, 'edge_weight', None) is not None:
                    aug.edge_weight = aug.edge_weight[keep_mask]
                if getattr(aug, 'edge_attr', None) is not None:
                    aug.edge_attr = aug.edge_attr[keep_mask]
            if aug.x is not None and feature_mask_rate > 0:
                feat_mask = torch.rand_like(aug.x) < feature_mask_rate
                aug.x = aug.x.masked_fill(feat_mask, 0.0)
            if aug.x is not None and noise_std > 0:
                aug.x = aug.x + torch.randn_like(aug.x) * noise_std
            graph_list.append(aug)
        return Batch.from_data_list(graph_list).to(self.config.device)

    def build_contrastive_views(self, data: Batch):
        """Build two views for contrastive learning (SimCSE-style).

        Both views use the SAME graph. The only difference comes from
        dropout masks during forward passes. This is the simplest possible
        contrastive setup — if this doesn't work, the architecture has issues.

        Granular-ball cross-scale alignment is added as auxiliary loss.
        """
        graph_list = data.to_data_list()
        view1 = Batch.from_data_list([g.clone() for g in graph_list]).to(self.config.device)
        # View2 = same graph, different dropout mask during forward()
        view2 = Batch.from_data_list([g.clone() for g in graph_list]).to(self.config.device)

        return view1, view2

    def _build_granular_ball_view(self, data: Batch):
        """Build granular-ball coarse view for cross-scale auxiliary loss."""
        from GOOD.data.gb import build_granular_ball_view
        graph_list = data.to_data_list()
        ball_r = getattr(self.config.train, 'ball_r', 0.5)
        coarse_graphs = [build_granular_ball_view(g, ball_r=ball_r) for g in graph_list]
        return Batch.from_data_list(coarse_graphs).to(self.config.device)

    def _compute_pretrain_loss(self, data: Batch) -> torch.Tensor:
        # === Primary: same-scale contrastive loss (original vs augmented) ===
        view1, view2 = self.build_contrastive_views(data)
        out1 = self.model(data=view1, ood_algorithm=self.ood_algorithm)
        out2 = self.model(data=view2, ood_algorithm=self.ood_algorithm)
        raw_pred = (out1, out2)
        targets = getattr(data, 'y', None)
        # Use SimCLR (labels=None) during pretrain: binary SupCon is degenerate
        # (too many positives per sample -> diffuse signal -> pair_sim drops)
        contrastive_loss = self.ood_algorithm.loss_calculate(raw_pred, None, None, None, self.config)

        # === Auxiliary: cross-scale alignment (BYOL-style, stop-gradient on coarse) ===
        gb_weight = getattr(self.config.train, 'gb_weight', 0.5)
        if gb_weight > 0:
            try:
                view_gb = self._build_granular_ball_view(data)
                out_gb = self.model(data=view_gb, ood_algorithm=self.ood_algorithm)
                # Stop-gradient on granular ball side — don't require GNN to produce
                # identical embeddings for structurally different graphs
                out_gb_sg = out_gb.detach()
                # Cosine similarity alignment loss (MSE on normalized embeddings)
                z1 = torch.nn.functional.normalize(out1, dim=1)
                z_gb = torch.nn.functional.normalize(out_gb_sg, dim=1)
                gb_loss = 2.0 - 2.0 * (z1 * z_gb).sum(dim=1).mean()
                contrastive_loss = contrastive_loss + gb_weight * gb_loss
            except Exception:
                pass  # Skip if granular ball fails

        return contrastive_loss

    def _train_pretrain_microbatches(self, data: Batch, pretrain_bs: int) -> dict:
        graph_list = data.to_data_list()
        total_graphs = len(graph_list)
        self.ood_algorithm.optimizer.zero_grad()

        aggregated_loss = torch.zeros((), device=self.config.device)
        processed_graphs = 0

        for start in range(0, total_graphs, pretrain_bs):
            chunk_graphs = graph_list[start:start + pretrain_bs]
            if len(chunk_graphs) < 2:
                continue

            chunk = Batch.from_data_list(chunk_graphs).to(self.config.device)
            chunk_loss = self._compute_pretrain_loss(chunk)
            chunk_weight = len(chunk_graphs) / total_graphs
            (chunk_loss * chunk_weight).backward()
            aggregated_loss = aggregated_loss + chunk_loss.detach() * chunk_weight
            processed_graphs += len(chunk_graphs)

            del chunk, chunk_loss
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()

        if processed_graphs == 0:
            zero_loss = torch.zeros((), device=self.config.device)
            self.ood_algorithm.mean_loss = zero_loss
            return {'loss': zero_loss}

        self.ood_algorithm.optimizer.step()
        self.ood_algorithm.mean_loss = aggregated_loss
        return {'loss': aggregated_loss.detach()}

    def train_batch(self, data: Batch, pbar) -> dict:
        r"""
        Train a batch. (Project use only)

        Args:
            data (Batch): Current batch of data.

        Returns:
            Calculated loss.
        """
        # 获取当前模式 (默认 finetune 以防未设置)
        current_mode = getattr(self.ood_algorithm, 'current_mode', 'finetune')

        # ==================================================
        # 分支 A: 预训练模式 (Contrastive Learning)
        # ==================================================
        if current_mode == 'pretrain':
            pretrain_bs = getattr(self.config.train, 'pretrain_bs', None)
            if pretrain_bs and getattr(data, 'num_graphs', 1) > pretrain_bs:
                return self._train_pretrain_microbatches(data, pretrain_bs)

            data = data.to(self.config.device)
            self.ood_algorithm.optimizer.zero_grad()
            loss = self._compute_pretrain_loss(data)

        # ==================================================
        # 分支 B: 微调/分类模式 (Original Logic)
        # ==================================================
        else:
            data = data.to(self.config.device)
            self.ood_algorithm.optimizer.zero_grad()
            # 1. 准备标签和掩码
            mask, targets = nan2zero_get_mask(data, 'train', self.config)
            node_norm = data.get('node_norm') if self.config.model.model_level == 'node' else None
            node_norm = node_norm.reshape(targets.shape) if node_norm is not None else None
            
            # 2. 数据预处理
            data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(
                data, targets, mask, node_norm, self.model.training, self.config
            )
            
            # 3. 获取边权重 (如果有)
            edge_weight = data.get('edge_weight') if hasattr(data, 'edge_weight') else data.get('edge_norm')

            # 4. 前向传播 (此时 model 会自动走 classifier)
            model_output = self.model(data=data, edge_weight=edge_weight, ood_algorithm=self.ood_algorithm)
            
            # 5. 后处理与损失
            raw_pred = self.ood_algorithm.output_postprocess(model_output)
            loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config)
        # ==================================================
        # 公共部分：反向传播
        # ==================================================
        # 处理 loss 格式 (如果是 dict 或其他格式)
        loss = self.ood_algorithm.loss_postprocess(loss, data, None, self.config)
        
        # Gradient clipping to stabilize pretrain
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.ood_algorithm.optimizer.step()

        return {'loss': loss.detach()}

    def train(self, fold=0):
        r"""
        Training pipeline. (Project use only)
        """

        # 1. 初始化配置
        self.config_model('train', fold)
        # 2. 设置为【预训练模式】
        self.ood_algorithm.set_up(self.model, self.config)
        # 现在是对比学习
        self.ood_algorithm.set_stage('pretrain',self.config)
        # ==========================================
        # 第一阶段：预训练 (Pre-training)
        # ==========================================
        pretrain_epochs = getattr(self.config.train, 'pre_epoch', 50)
        pretrain_bs = getattr(self.config.train, 'pretrain_bs', self.config.train.train_bs)
        print(f'#IN# Pre-train effective batch size: {pretrain_bs}')

        warmup_epochs = min(5, pretrain_epochs // 2)
        base_lr = self.config.train.pre_lr if hasattr(self.config.train, 'pre_lr') else 5e-4
        for epoch in range(pretrain_epochs):
            self.config.train.epoch = epoch
            # Linear warmup
            if epoch < warmup_epochs:
                warmup_lr = 1e-5 + (base_lr - 1e-5) * epoch / max(warmup_epochs, 1)
                for pg in self.ood_algorithm.optimizer.param_groups:
                    pg['lr'] = warmup_lr
            mean_loss = 0
            processed_steps = 0
            self.ood_algorithm.stage_control(self.config)

            pbar = tqdm(
                enumerate(self.loader['train']),
                total=len(self.loader['train']),
                desc=f'Pretrain {epoch + 1}/{pretrain_epochs}',
                leave=False,
            )
            for index, data in pbar:
                if data.batch is not None and (data.batch[-1] < self.config.train.train_bs - 1):
                    continue

                train_stat = self.train_batch(data, None)
                processed_steps += 1
                mean_loss = (mean_loss * (processed_steps - 1) + self.ood_algorithm.mean_loss) / processed_steps
                try:
                    pbar.set_postfix(loss=f'{float(mean_loss):.4f}')
                except Exception:
                    pass

            # Print diagnostics if available
            diag_str = ""
            if hasattr(self.ood_algorithm, '_diag_emb_std'):
                diag_str = (f" | emb_std={self.ood_algorithm._diag_emb_std:.4f}"
                            f" pair_sim={self.ood_algorithm._diag_pair_sim:.4f}"
                            f" neg_sim={self.ood_algorithm._diag_neg_sim:.4f}")
            print(f'#IN# Pre-train Epoch {epoch}: Loss {mean_loss:.4f}{diag_str}')
            
            # 预训练阶段通常不需要频繁做完整的 val/test 评估，或者只看 loss 即可
            # 如果想看 embedding 质量，可以加简单的评估

        # ==========================================
        # 切换到微调模式
        # ==========================================
        print(">>> [Snapshot] 正在保存预训练模型快照 (temp_pretrain_snapshot.pt)...")
        # 保存到当前目录下，方便读取
        torch.save(self.model.state_dict(), 'temp_pretrain_snapshot.pt')
        self.ood_algorithm.set_stage('finetune',self.config)
        finetune_epochs = getattr(self.config.train, 'ft_epochs', self.config.train.max_epoch)
        best_val_score = -1.0
        patience_counter = 0
        patience = getattr(self.config.train, 'patience', 15)
        # train the model
        for epoch in range(finetune_epochs):
            self.config.train.epoch = epoch
            # print(f'#IN#Epoch {epoch}:')

            mean_loss = 0
            spec_loss = 0

            self.ood_algorithm.stage_control(self.config)

            # pbar = tqdm(enumerate(self.loader['train']), total=len(self.loader['train']), **pbar_setting)
            # for index, data in pbar:
            for index, data in enumerate(self.loader['train']):
                if data.batch is not None and (data.batch[-1] < self.config.train.train_bs - 1):
                    continue

                # Parameter for DANN
                p = (index / len(self.loader['train']) + epoch) / finetune_epochs
                self.config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1
                # train a batch
                # train_stat = self.train_batch(data, pbar)
                train_stat = self.train_batch(data, None)
                mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (index + 1)

                if self.ood_algorithm.spec_loss is not None:
                    if isinstance(self.ood_algorithm.spec_loss, dict):
                        desc = f'ML: {mean_loss:.4f}|'
                        for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                            if not isinstance(spec_loss, dict):
                                spec_loss = dict()
                            if loss_name not in spec_loss.keys():
                                spec_loss[loss_name] = 0
                            spec_loss[loss_name] = (spec_loss[loss_name] * index + loss_value) / (index + 1)
                            desc += f'{loss_name}: {spec_loss[loss_name]:.4f}|'
                        # pbar.set_description(desc[:-1])
                    else:
                        spec_loss = (spec_loss * index + self.ood_algorithm.spec_loss) / (index + 1)
                        # pbar.set_description(f'M/S Loss: {mean_loss:.4f}/{spec_loss:.4f}')
                # else:
                #     pbar.set_description(f'Loss: {mean_loss:.4f}')

            # Eval training score

            # Epoch val
            # print('#IN#\nEvaluating...')
            if self.ood_algorithm.spec_loss is not None:
                if isinstance(self.ood_algorithm.spec_loss, dict):
                    desc = f'ML: {mean_loss:.4f}|'
                    for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                        desc += f'{loss_name}: {spec_loss[loss_name]:.4f}|'
                    print(f'#IN#Epoch {epoch}: Approximated ' + desc[:-1])
                else:
                    print(f'#IN#Epoch {epoch}: Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}')
            # else:
                # print(f'#IN#Epoch {epoch}: Approximated average training loss {mean_loss.cpu().item():.4f}')

            epoch_train_stat = self.evaluate('eval_train')
            id_val_stat = self.evaluate('id_val')
            id_test_stat = self.evaluate('id_test', True)
            val_stat = self.evaluate('val')
            test_stat = self.evaluate('test', True)
            print(f'#IN#Epoch {epoch}: Train acc {epoch_train_stat["score"]:.4f}, '
                  f'ID_val acc {id_val_stat["score"]:.4f}, '
                  f'ID_test acc {id_test_stat["score"]:.4f},'
                  f'OOD_val acc {val_stat["score"]:.4f}, '
                  f'OOD_test acc {test_stat["score"]:.4f}')
            # print(f'#IN#Epoch {epoch}: Test precision {test_stat["precision"]:.4f}, '
            #       f'recall {test_stat["recall"]:.4f}, f1 {test_stat["f1"]:.4f}, roc_auc {test_stat["roc_auc"]:.4f}\n')

            # checkpoints save
            self.save_epoch(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat, self.config, fold)

            # --- early stopping ---
            combined_val = (val_stat['score'] + id_val_stat['score']) / 2 if id_val_stat.get('score') else val_stat['score']
            if combined_val > best_val_score:
                best_val_score = combined_val
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f'#IN# Early stopping at epoch {epoch} (best val: {best_val_score:.4f})')
                break

            # --- scheduler step ---
            self.ood_algorithm.scheduler.step()

        # print('#IN#Training end.')

    @torch.no_grad()
    def evaluate(self, split: str, full_metrics: bool = False) -> Dict[str, float]:
        r"""
        This function is design to collect data results and calculate scores and loss given a dataset subset.
        (For project use only)

        Args:
            split (str): A split string for choosing the corresponding dataloader. Allowed: 'train', 'id_val', 'id_test',
                'val', and 'test'.

        Returns:
            A score and a loss.

        """
        stat = {'score': None, 'loss': None}
        if self.loader.get(split) is None:
            return stat
        self.model.eval()

        loss_all = []
        mask_all = []
        pred_all = []
        target_all = []
        # pbar = tqdm(self.loader[split], desc=f'Eval {split.capitalize()}', total=len(self.loader[split]),
        #             **pbar_setting)
        # for data in pbar:
        for data in self.loader[split]:
            data: Batch = data.to(self.config.device)

            mask, targets = nan2zero_get_mask(data, split, self.config)
            if mask is None:
                return stat
            node_norm = torch.ones_like(targets,
                                        device=self.config.device) if self.config.model.model_level == 'node' else None
            data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                                 self.model.training,
                                                                                 self.config)
            model_output = self.model(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            raw_preds = self.ood_algorithm.output_postprocess(model_output)

            # --------------- Loss collection ------------------
            loss: torch.tensor = self.config.metric.loss_func(raw_preds, targets, reduction='none') * mask
            mask_all.append(mask)
            loss_all.append(loss)

            # ------------- Score data collection ------------------
            pred, target = eval_data_preprocess(data.y, raw_preds, mask, self.config)
            pred_all.append(pred)
            target_all.append(target)

        # ------- Loss calculate -------
        loss_all = torch.cat(loss_all)
        mask_all = torch.cat(mask_all)
        stat['loss'] = loss_all.sum() / mask_all.sum()

        # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
        stat['score'] = eval_score(pred_all, target_all, self.config.metric.score_func)

        # ----------calculation of more metrics----------------
        if self.config.metric.dataset_task == 'Binary classification' and full_metrics:
            stat['precision'] = eval_score(pred_all, target_all, self.config.metric.precision)
            stat['recall'] = eval_score(pred_all, target_all, self.config.metric.recall)
            stat['f1'] = eval_score(pred_all, target_all, self.config.metric.f1)
            stat['roc_auc'] = eval_score(pred_all, target_all, self.config.metric.roc_auc_score)
        else:
            stat['precision'] = stat['score']
            stat['recall'] = stat['score']
            stat['f1'] = stat['score']
            stat['roc_auc'] = stat['score']

        # print(f'#IN#\n{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f}\n'
        #       f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

        self.model.train()

        return {'score': stat['score'], 'loss': stat['loss'], 'precision': stat['precision'],
                'recall': stat['recall'], 'f1': stat['f1'], 'roc_auc': stat['roc_auc'],
                'subject_num': mask_all.sum()}

    def load_task(self, fold=0):
        r"""
        Launch a training or a test.
        """
        if self.task == 'train':
            self.train(fold)

        elif self.task == 'test':

            # config model
            print('#D#Config model and output the best checkpoint info...')
            in_ckpt, ckpt = self.config_model('test', fold)
            return in_ckpt, ckpt

    def config_model(self, mode: str, load_param=False, fold=0):
        r"""
        A model configuration utility. Responsible for transiting model from CPU -> GPU and loading checkpoints.
        Args:
            mode (str): 'train' or 'test'.
            load_param: When True, loading test checkpoint will load parameters to the GNN model.

        Returns:
            Test score and loss if mode=='test'.
        """
        self.model.to(self.config.device)
        self.model.train()

        # load checkpoint
        if mode == 'train' and self.config.train.tr_ctn:
            ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'last{fold}.ckpt'))
            self.model.load_state_dict(ckpt['state_dict'])
            best_ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'best{fold}.ckpt'))
            self.config.metric.best_stat['score'] = best_ckpt['val_score']
            self.config.metric.best_stat['loss'] = best_ckpt['val_loss']
            self.config.train.ctn_epoch = ckpt['epoch'] + 1
            print(f'#IN#Continue training from Epoch {ckpt["epoch"]}...')

        if mode == 'test':
            try:
                ckpt = torch.load(self.config.test_ckpt, map_location=self.config.device)
            except FileNotFoundError:
                print(f'#E#Checkpoint not found at {os.path.abspath(self.config.test_ckpt)}')
                exit(1)
            if os.path.exists(self.config.id_test_ckpt):
                id_ckpt = torch.load(self.config.id_test_ckpt, map_location=self.config.device)
                # model.load_state_dict(id_ckpt['state_dict'])
                print(f'#IN#Loading best In-Domain Checkpoint {id_ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {id_ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {id_ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {id_ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {id_ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {id_ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {id_ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {id_ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {id_ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {id_ckpt["ood_val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {id_ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {id_ckpt["ood_test_loss"].item():.4f}\n')
                print(f'#IN#Loading best Out-of-Domain Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {ckpt["ood_val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {ckpt["ood_test_loss"].item():.4f}\n')

                print(f'#IN#ChartInfo {id_ckpt["id_test_score"]:.4f} {id_ckpt["test_score"]:.4f} '
                      f'{ckpt["id_test_score"]:.4f} {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')

            else:
                print(f'#IN#No In-Domain checkpoint.')
                # model.load_state_dict(ckpt['state_dict'])
                print(f'#IN#Loading best Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(
                    f'#IN#ChartInfo {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')
            # if load_param:
            #     if self.config.ood.ood_alg != 'EERM':
            #         self.model.load_state_dict(ckpt['state_dict'])
            #     else:
            #         self.model.gnn.load_state_dict(ckpt['state_dict'])
            # return ckpt["test_score"], ckpt["test_loss"]
            if load_param:
                self.model.load_state_dict(ckpt['state_dict'])
            return id_ckpt, ckpt

    def save_epoch(self, epoch: int, train_stat: dir, id_val_stat: dir, id_test_stat: dir, val_stat: dir,
                   test_stat: dir, config: Union[CommonArgs, Munch], fold=0):
        r"""
        Training util for checkpoint saving.

        Args:
            epoch (int): epoch number
            train_stat (dir): train statistics
            id_val_stat (dir): in-domain validation statistics
            id_test_stat (dir): in-domain test statistics
            val_stat (dir): ood validation statistics
            test_stat (dir): ood test statistics
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.ckpt_dir`, :obj:`config.dataset`, :obj:`config.train`, :obj:`config.model`, :obj:`config.metric`, :obj:`config.log_path`, :obj:`config.ood`)

        Returns:
            None

        """
        state_dict = self.model.state_dict() 
        # state_dict = self.model.state_dict() if config.ood.ood_alg != 'EERM' else self.model.gnn.state_dict()
        ckpt = {
            'state_dict': state_dict,
            'train_score': train_stat['score'],
            'train_loss': train_stat['loss'],
            'id_val_score': id_val_stat['score'],
            'id_val_loss': id_val_stat['loss'],
            'id_test_score': id_test_stat['score'],
            'id_test_loss': id_test_stat['loss'],
            'id_test_precision': id_test_stat['precision'],
            'id_test_recall': id_test_stat['recall'],
            'id_test_f1': id_test_stat['f1'],
            'id_test_roc_auc': id_test_stat['roc_auc'],
            'ood_val_score': val_stat['score'],
            'ood_val_loss': val_stat['loss'],
            'ood_test_score': test_stat['score'],
            'ood_test_loss': test_stat['loss'],
            'ood_test_precision': test_stat['precision'],
            'ood_test_recall': test_stat['recall'],
            'ood_test_f1': test_stat['f1'],
            'ood_test_roc_auc': test_stat['roc_auc'],
            'val_score': (val_stat['score'] * val_stat['subject_num'] + id_val_stat['score'] * id_val_stat['subject_num']) / (
                    val_stat['subject_num'] + id_val_stat['subject_num']),
            'test_score': (test_stat['score'] * test_stat['subject_num'] + id_test_stat['score'] * id_test_stat['subject_num']) / (
                    test_stat['subject_num'] + id_test_stat['subject_num']),
            'id_val_subject_num': id_val_stat['subject_num'],
            'id_test_subject_num': id_test_stat['subject_num'],
            'ood_val_subject_num': val_stat['subject_num'],
            'ood_test_subject_num': test_stat['subject_num'],
            'time': datetime.datetime.now().strftime('%b%d %Hh %M:%S'),
            'model': {
                'model name': f'{config.model.model_name} {config.model.model_level} layers',
                'dim_hidden': config.model.dim_hidden,
                'dim_ffn': config.model.dim_ffn,
                'global pooling': config.model.global_pool
            },
            'dataset': config.dataset.dataset_name,
            'train': {
                'weight_decay': config.train.weight_decay,
                'learning_rate': config.train.lr,
                'mile stone': config.train.mile_stones,
                'shift_type': config.dataset.shift_type,
                'Batch size': f'{config.train.train_bs}, {config.train.val_bs}, {config.train.test_bs}'
            },
            'OOD': {
                'OOD alg': config.ood.ood_alg,
                'OOD param': config.ood.ood_param,
                'number of environments': config.dataset.num_envs
            },
            'log file': config.log_path,
            'epoch': epoch,
            'max epoch': config.train.max_epoch
        }
        if not (config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat[
            'score'] < config.metric.lower_better *
                config.metric.best_stat['score']
                or (id_val_stat.get('score') and (
                        config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
                    'score'] < config.metric.lower_better * config.metric.id_best_stat['score']))
                or epoch % config.train.save_gap == 0):
            return

        if not os.path.exists(config.ckpt_dir):
            os.makedirs(config.ckpt_dir)
            print(f'#W#Directory does not exists. Have built it automatically.\n'
                  f'{os.path.abspath(config.ckpt_dir)}')
        saved_file = os.path.join(config.ckpt_dir, f'{epoch}.ckpt')
        torch.save(ckpt, saved_file)
        shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'last{fold}.ckpt'))

        # --- In-Domain checkpoint ---
        if id_val_stat.get('score') and (
                config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
            'score'] < config.metric.lower_better * config.metric.id_best_stat['score']):
            config.metric.id_best_stat['score'] = id_val_stat['score']
            config.metric.id_best_stat['loss'] = id_val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'id_best{fold}.ckpt'))
            print('#IM#Saved a new best In-Domain checkpoint.\n')

        # --- Out-Of-Domain checkpoint ---
        # if id_val_stat.get('score'):
        #     if not (config.metric.lower_better * id_val_stat['score'] < config.metric.lower_better * val_stat['score']):
        #         return
        if config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat[
            'score'] < config.metric.lower_better * \
                config.metric.best_stat['score']:
            config.metric.best_stat['score'] = val_stat['score']
            config.metric.best_stat['loss'] = val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'best{fold}.ckpt'))
            print('#IM#Saved a new best checkpoint.\n')
        if config.clean_save:
            os.unlink(saved_file)
