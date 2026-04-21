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



    def train_batch(self, data: Batch, pbar) -> dict:
        r"""
        Train a batch: classification + GBCR losses.

        GBCR is embedded in the GNN forward pass (between GAT layers).
        After forward, we extract GBCR info for auxiliary losses:
          - L_align: class-conditional ball importance alignment across sites
          - L_entropy: assignment entropy (balanced balls)
          - L_site: site adversarial on calibrated features

        Staged training (max_epoch=35, early_stop from epoch 25):
        Stage 1 (epoch < 8):   cls + gate losses (SiteCalibration learns freely)
        Stage 2 (8 <= epoch < 20): + SiteAdv (weight=0.05)
        Stage 3 (18 <= epoch < 25): + GBCR (7 epochs of training)
        Stage 4 (epoch >= 25):  all aux frozen, backbone-only finetune
        Early stopping only allowed from epoch 25+; ckpt candidates from epoch 10+.
        """
        data = data.to(self.config.device)
        self.ood_algorithm.optimizer.zero_grad()

        # --- 1. Forward pass (GBCR runs inside GNN automatically) ---
        mask, targets = nan2zero_get_mask(data, 'train', self.config)
        node_norm = data.get('node_norm') if self.config.model.model_level == 'node' else None
        node_norm = node_norm.reshape(targets.shape) if node_norm is not None else None

        data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(
            data, targets, mask, node_norm, self.model.training, self.config
        )

        edge_weight = data.get('edge_weight') if hasattr(data, 'edge_weight') else data.get('edge_norm')
        model_output = self.model(data=data, edge_weight=edge_weight, ood_algorithm=self.ood_algorithm)

        raw_pred = self.ood_algorithm.output_postprocess(model_output)
        cls_loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config)
        cls_loss = self.ood_algorithm.loss_postprocess(cls_loss, data, mask, self.config)

        # --- 2. Staged auxiliary losses ---
        epoch = self.config.train.epoch
        site_adv_warmup = getattr(self.config.train, 'gate_warmup_epoch', 8)
        site_adv_stop = getattr(self.config.train, 'site_adv_stop_epoch', 20)
        gbcr_warmup = getattr(self.config.train, 'gbcr_warmup_epoch', 18)
        gbcr_stop = getattr(self.config.train, 'gbcr_stop_epoch', 25)

        align_val = 0.0
        entropy_val = 0.0
        site_adv_val = 0.0
        gate_align_val = 0.0

        # --- Stage 1: gate regularization (delayed to let gate learn first) ---
        gate_reg_warmup = getattr(self.config.train, 'gate_reg_warmup', 8)
        calib_info = self.model.calib_info
        if calib_info:
            if epoch >= gate_reg_warmup:
                # Linear warmup over 5 epochs after warmup starts
                ramp = min(1.0, (epoch - gate_reg_warmup + 1) / 5.0)

                # L_aff = ||gamma-1||^2 + ||beta||^2 (keeps affine mild)
                affine_w = getattr(self.config.ood, 'affine_reg_weight', 0.005) * ramp
                cls_loss = cls_loss + affine_w * calib_info['affine_reg']

                # L_gate-range: penalize if gate mean outside [0.05, 0.15]
                gate_sp_w = getattr(self.config.ood, 'gate_sparsity_weight', 0.05) * ramp
                cls_loss = cls_loss + gate_sp_w * calib_info['gate_reg']

                # L_gate-align: EMA-smoothed alignment, also warmed up
                if hasattr(data, 'env_id') and data.env_id is not None and hasattr(data, 'y'):
                    gate_align_w = getattr(self.config.ood, 'gate_align_weight', 0.1) * ramp
                    y_graph = data.y.view(-1)
                    env_graph = data.env_id.view(-1)
                    num_graphs = calib_info['gate'].size(0)
                    if y_graph.size(0) == num_graphs and env_graph.size(0) == num_graphs:
                        gate_align_loss = self.model.site_calibration.gate_alignment_loss(
                            calib_info['gate'], None, y_graph, env_graph)
                        gate_align_val = gate_align_loss.item()
                        cls_loss = cls_loss + gate_align_w * gate_align_loss

        # --- Stage 2 (site_adv_warmup <= epoch < site_adv_stop): SiteAdv ---
        if site_adv_warmup <= epoch < site_adv_stop and getattr(data, 'num_graphs', 1) >= 2:
            site_adv_w = getattr(self.config.ood, 'site_adv_weight', 0.05)
            if site_adv_w > 0 and self.model.site_calibration.site_classifier is not None:
                if hasattr(data, 'env_id') and data.env_id is not None:
                    # Use calibrated features from encoder (already computed in forward)
                    encoder = self.model.gnn.encoder
                    if encoder._calib_info is not None:
                        batch_idx = data.batch if data.batch is not None else torch.zeros(
                            data.x.size(0), dtype=torch.long, device=data.x.device)
                        # Get post-GAT1+calib node features for site adversarial
                        # We reuse the encoder's stored calibrated output
                        # Actually run site_adv on the graph-level pooled calibrated features
                        grl_lam = self.config.train.alpha
                        # Forward through site_calibration was already done in encoder
                        # We need the calibrated H for site_adv — it's post_conv after calib
                        # For simplicity, re-run calibration on the stored H1
                        # But actually, the encoder already ran it. We use the model-level
                        # site_adversarial_loss which pools and classifies.
                        # We need to get H_calibrated from encoder. Let's add it.
                        if hasattr(encoder, '_h_calibrated'):
                            site_adv_loss = self.model.site_calibration.site_adversarial_loss(
                                encoder._h_calibrated, batch_idx, data.env_id, grl_lambda=grl_lam)
                            site_adv_val = site_adv_loss.item()
                            cls_loss = cls_loss + site_adv_w * site_adv_loss

        # --- Stage 3 (gbcr_warmup <= epoch < gbcr_stop): short GBCR warm-up window ---
        if gbcr_warmup <= epoch < gbcr_stop and getattr(data, 'num_graphs', 1) >= 2:
            # Freeze gate parameters
            for p in self.model.site_calibration.parameters():
                p.requires_grad = False

            gbcr_info = self.model.get_gbcr_info()
            if gbcr_info is not None:
                gbcr_module = self.model.gnn.encoder.gbcr

                # Class-conditional importance alignment across sites
                lambda_align = getattr(self.config.ood, 'gbcr_align_weight', 0.1)
                if hasattr(data, 'env_id') and data.env_id is not None and hasattr(data, 'y'):
                    align_loss_val = gbcr_module.importance_alignment_loss(
                        data.batch, data.y.view(-1), data.env_id)
                    if align_loss_val.item() > 0:
                        align_val = align_loss_val.item()
                        cls_loss = cls_loss + lambda_align * align_loss_val

                # Assignment entropy (balanced balls)
                lambda_entropy = getattr(self.config.ood, 'gbcr_entropy_weight', 0.05)
                entropy_loss = gbcr_module.assignment_entropy_loss()
                if entropy_loss.item() > 0:
                    entropy_val = entropy_loss.item()
                    cls_loss = cls_loss + lambda_entropy * entropy_loss

        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.ood_algorithm.optimizer.step()

        # Keep gate frozen from gbcr_warmup onward (site_calibration already static).
        if epoch >= gbcr_warmup:
            for p in self.model.site_calibration.parameters():
                p.requires_grad = False
        # Stage 4: after gbcr_stop, freeze GBCR so only backbone trains.
        if epoch >= gbcr_stop and hasattr(self.model.gnn, 'encoder') and hasattr(self.model.gnn.encoder, 'gbcr'):
            for p in self.model.gnn.encoder.gbcr.parameters():
                p.requires_grad = False

        # Store for logging
        self.ood_algorithm.spec_loss = {
            'Align': align_val,
            'Entropy': entropy_val,
            'SiteAdv': site_adv_val,
            'GateAlign': gate_align_val,
        }

        return {'loss': cls_loss.detach()}

    def train(self, fold=0):
        r"""
        Training with MA3-based checkpoint selection and top-3 ensemble.
        Stage 1+2 only for main results; stage 3 (GB/VICReg) ablation-only.
        """
        self.config_model('train', fold)
        self.ood_algorithm.set_up(self.model, self.config)
        self.ood_algorithm.set_stage('finetune', self.config)
        self.model.set_mode('finetune')

        max_epochs = self.config.train.max_epoch
        patience = getattr(self.config.train, 'patience', 8)

        # --- MA5 scoring + post-hoc NMS top-3 checkpoints ---
        score_history = []  # raw S_t per epoch
        top_k = 3
        min_peak_gap = 4  # temporal NMS radius (in epochs)
        # Track every saved epoch; NMS runs after training finishes.
        all_epoch_ckpts = []  # list of (epoch, snapshot_ma5, ckpt_path)

        patience_counter = 0
        best_patience_score = -1.0

        for epoch in range(max_epochs):
            self.config.train.epoch = epoch
            mean_loss = 0
            spec_loss = 0

            self.ood_algorithm.stage_control(self.config)

            last_trained_data = None
            for index, data in enumerate(self.loader['train']):
                if data.batch is not None and (data.batch[-1] < self.config.train.train_bs - 1):
                    continue
                p = (index / len(self.loader['train']) + epoch) / max_epochs
                self.config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1
                train_stat = self.train_batch(data, None)
                last_trained_data = data  # aligns with encoder._h_pre_gate / _h_calibrated
                mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (index + 1)

                if self.ood_algorithm.spec_loss is not None:
                    if isinstance(self.ood_algorithm.spec_loss, dict):
                        if not isinstance(spec_loss, dict):
                            spec_loss = dict()
                        for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                            if loss_name not in spec_loss:
                                spec_loss[loss_name] = 0
                            spec_loss[loss_name] = (spec_loss[loss_name] * index + loss_value) / (index + 1)

            # Print training loss
            if self.ood_algorithm.spec_loss is not None and isinstance(spec_loss, dict):
                desc = f'ML: {mean_loss:.4f}|'
                for ln, lv in spec_loss.items():
                    desc += f'{ln}: {lv:.4f}|'
                print(f'#IN#Epoch {epoch}: {desc[:-1]}')

            # Gate monitoring
            gate_info = self.model.calib_info if hasattr(self.model, 'calib_info') and self.model.calib_info else {}
            if gate_info:
                print(f'#IN#  Gate: mean={gate_info.get("gate_mean",0):.4f} '
                      f'std={gate_info.get("gate_std",0):.4f} '
                      f'||\u03b3-1||={gate_info.get("gamma_dev",0):.4f} '
                      f'||\u03b2||={gate_info.get("beta_norm",0):.4f}')

            # Site probe: pre-gate vs post-gate site classification accuracy.
            # IMPORTANT: use last_trained_data — it is the exact batch whose forward
            # produced encoder._h_pre_gate / _h_calibrated. Re-iterating the loader
            # would misalign batch/env_id against the cached representations.
            encoder = self.model.gnn.encoder if hasattr(self.model.gnn, 'encoder') else None
            if (encoder is not None and hasattr(encoder, '_h_pre_gate')
                    and encoder._h_pre_gate is not None and encoder._h_calibrated is not None
                    and last_trained_data is not None
                    and hasattr(last_trained_data, 'batch')
                    and hasattr(last_trained_data, 'env_id')
                    and last_trained_data.env_id is not None):
                try:
                    probe_batch = last_trained_data.batch.to(self.config.device)
                    probe_env = last_trained_data.env_id.view(-1).to(self.config.device)
                    # Sanity check: pooled graph count must match env_id count
                    n_graphs_feat = int(probe_batch.max().item()) + 1
                    if n_graphs_feat != probe_env.size(0):
                        print(f'#WARN# SiteProbe skipped: n_graphs({n_graphs_feat}) != env_id({probe_env.size(0)})')
                    else:
                        pre_acc, post_acc = self.model.site_calibration.site_probe_accuracy(
                            encoder._h_pre_gate, encoder._h_calibrated,
                            probe_batch, probe_env)
                        print(f'#IN#  SiteProbe: pre={pre_acc:.4f} post={post_acc:.4f}')
                except Exception as e:
                    print(f'#WARN# SiteProbe failed: {type(e).__name__}: {e}')

            # Evaluate
            epoch_train_stat = self.evaluate('eval_train')
            id_val_stat = self.evaluate('id_val', True)
            id_test_stat = self.evaluate('id_test', True)
            val_stat = self.evaluate('val', True)
            test_stat = self.evaluate('test', True)
            print(f'#IN#Epoch {epoch}: Train {epoch_train_stat["score"]:.4f}, '
                  f'ID_val {id_val_stat["score"]:.4f}(BA={id_val_stat.get("balanced_accuracy", 0):.4f}), '
                  f'ID_test {id_test_stat["score"]:.4f}, '
                  f'OOD_val {val_stat["score"]:.4f}(BA={val_stat.get("balanced_accuracy", 0):.4f}, AUC={val_stat.get("roc_auc", 0):.4f}), '
                  f'OOD_test {test_stat["score"]:.4f}(BA={test_stat.get("balanced_accuracy", 0):.4f})')

            # --- S_t with train-test gap penalty ---
            # Penalise source-overfit: if train BA >> val BA, downweight this epoch.
            ba_ood = val_stat.get('balanced_accuracy', val_stat['score']) or 0.0
            auroc_ood = val_stat.get('roc_auc', val_stat['score']) or 0.0
            ba_id = id_val_stat.get('balanced_accuracy', id_val_stat['score']) or 0.0
            ba_train = epoch_train_stat.get('balanced_accuracy', epoch_train_stat['score']) or 0.0
            ba_val_mix = 0.5 * ba_ood + 0.5 * ba_id
            gap_penalty = max(0.0, ba_train - ba_val_mix)
            s_t = 0.5 * auroc_ood + 0.2 * ba_ood + 0.2 * ba_id - 0.1 * gap_penalty
            score_history.append(s_t)

            # MA5: average of last 5 S_t values
            window = score_history[-5:]
            ma5 = sum(window) / len(window)

            # --- Save checkpoint (always save for top-k tracking) ---
            ckpt = self._build_ckpt(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat)
            if not os.path.exists(self.config.ckpt_dir):
                os.makedirs(self.config.ckpt_dir)
            ckpt_path = os.path.join(self.config.ckpt_dir, f'{epoch}.ckpt')
            torch.save(ckpt, ckpt_path)
            shutil.copy(ckpt_path, os.path.join(self.config.ckpt_dir, f'last{fold}.ckpt'))

            # Record candidates for final ensemble (only mature epochs).
            # Epoch < 10 checkpoints are saved to disk for analysis but excluded
            # from the final NMS/ensemble — they are pure stage-1 and would
            # dominate over later, better-generalizing checkpoints.
            ckpt_min_epoch = getattr(self.config.train, 'ckpt_min_epoch', 10)
            if epoch >= ckpt_min_epoch:
                all_epoch_ckpts.append((epoch, ma5, ckpt_path))
            print(f'#IN#  MA5={ma5:.4f} S_t={s_t:.4f}')


            # --- Early stopping (only after all stages have had time to train) ---
            early_stop_start = getattr(self.config.train, 'early_stop_start_epoch', 25)
            if epoch >= early_stop_start:
                if ma5 > best_patience_score:
                    best_patience_score = ma5
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f'#IN# Early stopping at epoch {epoch} (best MA5: {best_patience_score:.4f})')
                    break
            else:
                # Before early_stop_start: still track best score but never stop
                if ma5 > best_patience_score:
                    best_patience_score = ma5

            self.ood_algorithm.scheduler.step()

        # --- Post-hoc: recompute trailing MA5 with full history, then NMS ---
        def _trailing_ma5(i):
            w = score_history[max(0, i - 4): i + 1]
            return sum(w) / len(w) if w else -1.0

        reranked = [(e, _trailing_ma5(e), p) for e, _m, p in all_epoch_ckpts
                    if os.path.exists(p)]
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Greedy temporal NMS: accept highest MA5, skip anything within min_peak_gap.
        # If fewer than top_k selected, relax gap from 4 → 3 and retry.
        def _nms(candidates, gap, k):
            selected = []
            for (e, m, p) in candidates:
                if len(selected) >= k:
                    break
                if any(abs(e - pe) < gap for (_, pe, _) in selected):
                    continue
                selected.append((m, e, p))
            return selected

        peak_ckpts = _nms(reranked, min_peak_gap, top_k)
        if len(peak_ckpts) < top_k and min_peak_gap > 3:
            peak_ckpts = _nms(reranked, 3, top_k)
            if len(peak_ckpts) > len(_nms(reranked, min_peak_gap, top_k)):
                print(f'#IN#  NMS relaxed gap {min_peak_gap}→3 to get {len(peak_ckpts)} checkpoints')
        print(f'#IN# NMS selected top-{len(peak_ckpts)} from {len(reranked)} candidates: '
              f'{[(e, f"{m:.4f}") for m, e, _ in peak_ckpts]}')

        # Delete non-selected ckpts (keep last{fold}.ckpt).
        kept_paths = {p for _, _, p in peak_ckpts}
        kept_paths.add(os.path.join(self.config.ckpt_dir, f'last{fold}.ckpt'))
        for (_, _, p) in reranked:
            if p not in kept_paths and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass

        # --- Save top-k info for test-time ensemble ---
        topk_info_path = os.path.join(self.config.ckpt_dir, f'topk{fold}.info')
        with open(topk_info_path, 'w') as f:
            for m, e, p in peak_ckpts:
                if os.path.exists(p):
                    f.write(f'{p}\t{m:.6f}\n')
        topk_paths = [p for _, _, p in peak_ckpts if os.path.exists(p)]
        print(f'#IN# Saved top-{len(topk_paths)} peak checkpoints for ensemble.')

        # --- Save best{fold}.ckpt and id_best{fold}.ckpt for backward compat ---
        if peak_ckpts:
            best_path = peak_ckpts[0][2]  # highest MA5
            shutil.copy(best_path, os.path.join(self.config.ckpt_dir, f"best{fold}.ckpt"))
            shutil.copy(best_path, os.path.join(self.config.ckpt_dir, f"id_best{fold}.ckpt"))
            print(f"#IN# Best checkpoint: epoch {peak_ckpts[0][1]} (MA5={peak_ckpts[0][0]:.4f})")

    @torch.no_grad()
    def evaluate(self, split: str, full_metrics: bool = True) -> Dict[str, float]:
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
            # Squeeze for cross_entropy compatibility
            t = targets.squeeze(-1) if targets.dim() > 1 and targets.shape[-1] == 1 else targets
            m = mask.squeeze(-1) if mask.dim() > 1 and mask.shape[-1] == 1 else mask
            loss: torch.tensor = self.config.metric.loss_func(raw_preds, t, reduction='none') * m
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
        if full_metrics:
            stat['precision'] = eval_score(pred_all, target_all, self.config.metric.precision)
            stat['recall'] = eval_score(pred_all, target_all, self.config.metric.recall)
            stat['f1'] = eval_score(pred_all, target_all, self.config.metric.f1)
            stat['roc_auc'] = eval_score(pred_all, target_all, self.config.metric.roc_auc_score)
            stat['balanced_accuracy'] = eval_score(pred_all, target_all, self.config.metric.balanced_accuracy)
        else:
            stat['precision'] = stat['score']
            stat['recall'] = stat['score']
            stat['f1'] = stat['score']
            stat['roc_auc'] = stat['score']
            stat['balanced_accuracy'] = stat['score']

        # print(f'#IN#\n{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f}\n'
        #       f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

        self.model.train()

        return {'score': stat['score'], 'loss': stat['loss'], 'precision': stat['precision'],
                'recall': stat['recall'], 'f1': stat['f1'], 'roc_auc': stat['roc_auc'],
                'balanced_accuracy': stat['balanced_accuracy'], 'subject_num': mask_all.sum()}

    def load_task(self, fold=0):
        r"""
        Launch a training or a test.
        """
        if self.task == 'train':
            self.train(fold)

        elif self.task == 'test':
            # Decision 3: ensemble top-k checkpoints
            topk_info_path = os.path.join(self.config.ckpt_dir, f'topk{fold}.info')
            if os.path.exists(topk_info_path):
                ckpt_paths = []
                ckpt_ma5s = []
                with open(topk_info_path) as f:
                    for l in f:
                        parts = l.strip().split('\t')
                        if not parts[0] or not os.path.exists(parts[0]):
                            continue
                        ckpt_paths.append(parts[0])
                        ckpt_ma5s.append(float(parts[1]) if len(parts) > 1 else 1.0)
                if ckpt_paths:
                    print(f'#IN# Ensemble evaluation with {len(ckpt_paths)} checkpoints')
                    ensemble_ckpt, single_ckpt = self._ensemble_evaluate(
                        ckpt_paths, ckpt_ma5s, fold)
                    return ensemble_ckpt, single_ckpt

            # Fallback to old behavior
            print('#D#Config model and output the best checkpoint info...')
            in_ckpt, ckpt = self.config_model('test', fold)
            return in_ckpt, ckpt

    @torch.no_grad()
    def _ensemble_evaluate(self, ckpt_paths, ma5_scores, fold):
        """Ensemble top-k checkpoints: MA5-weighted softmax probabilities."""
        from GOOD.utils.evaluation import eval_data_preprocess, eval_score

        all_probs = {}  # split -> list of prob arrays per checkpoint
        targets_cache = {}
        env_cache = {}      # split -> list of env_id tensors (for pseudo-OOD threshold tuning)
        masks_cache = {}

        for ci, cp in enumerate(ckpt_paths):
            ckpt = torch.load(cp, map_location=self.config.device)
            self.model.load_state_dict(ckpt['state_dict'])
            self.model.eval()
            print(f'  Checkpoint {ci}: epoch {ckpt["epoch"]}')

            for split in ['eval_train', 'id_val', 'id_test', 'val', 'test']:
                if self.loader.get(split) is None:
                    continue
                preds_this = []
                targets_this = []
                for data in self.loader[split]:
                    data = data.to(self.config.device)
                    mask, targets = nan2zero_get_mask(data, split, self.config)
                    if mask is None:
                        continue
                    data, targets, mask, _ = self.ood_algorithm.input_preprocess(
                        data, targets, mask, None, False, self.config)
                    out = self.model(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
                    raw = self.ood_algorithm.output_postprocess(out)
                    # Get softmax probs
                    probs = torch.softmax(raw, dim=1).cpu()
                    preds_this.append(probs)
                    if ci == 0:
                        if split not in targets_cache:
                            targets_cache[split] = []
                        targets_cache[split].append(data.y.cpu())
                        if hasattr(data, 'env_id') and data.env_id is not None:
                            if split not in env_cache:
                                env_cache[split] = []
                            env_cache[split].append(data.env_id.view(-1).cpu())

                if split not in all_probs:
                    all_probs[split] = []
                all_probs[split].append(torch.cat(preds_this, dim=0))

        # MA5-weighted softmax average (τ=5) across checkpoints
        tau = getattr(self.config.ood, 'ensemble_tau', 5.0)
        ma5_t = torch.tensor(ma5_scores, dtype=torch.float)
        ens_weights = torch.softmax(tau * ma5_t, dim=0)
        print(f'  Ensemble weights (τ={tau}): {[f"{w:.3f}" for w in ens_weights.tolist()]}')

        avg_probs_dict = {}
        targets_dict = {}
        env_dict = {}
        for split in all_probs:
            prob_stack = torch.stack(all_probs[split])  # [K, N, C]
            avg_probs_dict[split] = (prob_stack * ens_weights.view(-1, 1, 1)).sum(dim=0)
            targets_dict[split] = torch.cat(targets_cache.get(split, []), dim=0).squeeze()
            if env_cache.get(split):
                env_dict[split] = torch.cat(env_cache[split], dim=0)

        # --- Disagreement check: if top-k predictions diverge, fall back to single ---
        use_single = False
        if len(ckpt_paths) > 1 and 'val' in all_probs:
            prob_stack_val = torch.stack(all_probs['val'])  # [K, N, C]
            D = prob_stack_val[:, :, 1].var(dim=0).mean().item()
            if D > 0.02:
                print(f'  Disagreement D={D:.4f} > 0.02 → falling back to single-best probs')
                use_single = True
                for split in avg_probs_dict:
                    avg_probs_dict[split] = all_probs[split][0]  # highest MA5
            else:
                print(f'  Disagreement D={D:.4f} <= 0.02 → using ensemble')

        # --- Threshold tuning: blend 0.5*BA_OOD_val + 0.5*BA_ID_val ---
        from sklearn.metrics import balanced_accuracy_score as ba_score
        best_threshold = 0.5
        best_score = -1.0
        thr_grid = [0.40, 0.45, 0.50]

        for thr in thr_grid:
            ba_sources = []
            for src in ['val', 'id_val']:  # OOD_val + ID_val
                if src not in avg_probs_dict:
                    continue
                p1 = avg_probs_dict[src][:, 1]
                tg = targets_dict[src]
                pr = (p1 >= thr).long()
                ba_sources.append(ba_score(tg.numpy(), pr.numpy()))
            if ba_sources:
                score = sum(ba_sources) / len(ba_sources)
                if score > best_score:
                    best_score = score
                    best_threshold = thr
        print(f'  Threshold (0.5*BA_OOD + 0.5*BA_ID): {best_threshold:.3f} (score={best_score:.4f})')

        # Compute metrics with tuned threshold
        result_ckpt = {}
        for split in avg_probs_dict:
            avg_probs = avg_probs_dict[split]
            targets = targets_dict[split]
            prob_class1 = avg_probs[:, 1]
            preds = (prob_class1 >= best_threshold).long()
            acc = (preds == targets).float().mean().item()
            from sklearn.metrics import balanced_accuracy_score as ba_score
            from sklearn.metrics import roc_auc_score as sk_auroc
            from sklearn.metrics import precision_score, recall_score, f1_score
            ba = ba_score(targets.numpy(), preds.numpy())
            prec = precision_score(targets.numpy(), preds.numpy(), zero_division=0)
            rec = recall_score(targets.numpy(), preds.numpy(), zero_division=0)
            f1 = f1_score(targets.numpy(), preds.numpy(), zero_division=0)
            try:
                auroc = sk_auroc(targets.numpy(), prob_class1.numpy())
            except ValueError:
                auroc = 0.5

            prefix = {'eval_train': 'train', 'id_val': 'id_val', 'id_test': 'id_test',
                       'val': 'ood_val', 'test': 'ood_test'}.get(split, split)
            result_ckpt[f'{prefix}_score'] = acc
            result_ckpt[f'{prefix}_balanced_accuracy'] = ba
            result_ckpt[f'{prefix}_roc_auc'] = auroc
            result_ckpt[f'{prefix}_precision'] = prec
            result_ckpt[f'{prefix}_recall'] = rec
            result_ckpt[f'{prefix}_f1'] = f1
            print(f'  Ensemble {split}: acc={acc:.4f}, BA={ba:.4f}, AUROC={auroc:.4f} (thr={best_threshold:.2f})')

        # Fill in required fields for backward compat
        result_ckpt['epoch'] = 'ensemble'
        result_ckpt['train_score'] = result_ckpt.get('train_score', 0)
        result_ckpt['train_loss'] = torch.tensor(0.0)
        for key in ['id_val_loss', 'id_test_loss', 'ood_val_loss', 'ood_test_loss']:
            result_ckpt[key] = torch.tensor(0.0)
        result_ckpt['val_score'] = result_ckpt.get('ood_val_score', 0)
        result_ckpt['test_score'] = result_ckpt.get('ood_test_score', 0)
        # precision/recall/f1/roc_auc computed properly per split above
        result_ckpt['id_val_subject_num'] = len(targets_dict.get('id_val', []))
        result_ckpt['id_test_subject_num'] = len(targets_dict.get('id_test', []))
        result_ckpt['ood_val_subject_num'] = len(targets_dict.get('val', []))
        result_ckpt['ood_test_subject_num'] = len(targets_dict.get('test', []))

        # --- Single-best evaluation (same tuned threshold) ---
        single_result = {}
        for split in avg_probs_dict:
            single_probs = all_probs[split][0]  # first = highest MA5
            targets = targets_dict[split]
            prob_class1 = single_probs[:, 1]
            preds = (prob_class1 >= best_threshold).long()
            acc = (preds == targets).float().mean().item()
            ba = ba_score(targets.numpy(), preds.numpy())
            prec = precision_score(targets.numpy(), preds.numpy(), zero_division=0)
            rec = recall_score(targets.numpy(), preds.numpy(), zero_division=0)
            f1_val = f1_score(targets.numpy(), preds.numpy(), zero_division=0)
            try:
                auroc = sk_auroc(targets.numpy(), prob_class1.numpy())
            except ValueError:
                auroc = 0.5

            prefix = {'eval_train': 'train', 'id_val': 'id_val', 'id_test': 'id_test',
                       'val': 'ood_val', 'test': 'ood_test'}.get(split, split)
            single_result[f'{prefix}_score'] = acc
            single_result[f'{prefix}_balanced_accuracy'] = ba
            single_result[f'{prefix}_roc_auc'] = auroc
            single_result[f'{prefix}_precision'] = prec
            single_result[f'{prefix}_recall'] = rec
            single_result[f'{prefix}_f1'] = f1_val
            print(f'  Single  {split}: acc={acc:.4f}, BA={ba:.4f}, AUROC={auroc:.4f}')

        # Backward compat fields for single_result
        single_result['epoch'] = 'single_best'
        single_result['train_score'] = single_result.get('train_score', 0)
        single_result['train_loss'] = torch.tensor(0.0)
        for key in ['id_val_loss', 'id_test_loss', 'ood_val_loss', 'ood_test_loss']:
            single_result[key] = torch.tensor(0.0)
        single_result['val_score'] = single_result.get('ood_val_score', 0)
        single_result['test_score'] = single_result.get('ood_test_score', 0)
        single_result['id_val_subject_num'] = len(targets_dict.get('id_val', []))
        single_result['id_test_subject_num'] = len(targets_dict.get('id_test', []))
        single_result['ood_val_subject_num'] = len(targets_dict.get('val', []))
        single_result['ood_test_subject_num'] = len(targets_dict.get('test', []))

        return result_ckpt, single_result

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
                      f'OOD Validation {self.config.metric.score_name}: {id_ckpt["ood_val_score"]:.4f}\n'
                      f'OOD Validation Loss: {id_ckpt["ood_val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {id_ckpt["ood_test_score"]:.4f}\n'
                      f'OOD Test Loss: {id_ckpt["ood_test_loss"].item():.4f}\n'
                      f'Mixed(ID+OOD) Val: {id_ckpt["val_score"]:.4f}, Test: {id_ckpt["test_score"]:.4f}\n')
                print(f'#IN#Loading best Out-of-Domain Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {ckpt["ood_val_score"]:.4f}\n'
                      f'OOD Validation Loss: {ckpt["ood_val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {ckpt["ood_test_score"]:.4f}\n'
                      f'OOD Test Loss: {ckpt["ood_test_loss"].item():.4f}\n'
                      f'Mixed(ID+OOD) Val: {ckpt["val_score"]:.4f}, Test: {ckpt["test_score"]:.4f}\n')

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

    def _build_ckpt(self, epoch, train_stat, id_val_stat, id_test_stat, val_stat, test_stat):
        """Build checkpoint dict without saving."""
        config = self.config
        return {
            'state_dict': self.model.state_dict(),
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
            'ood_val_balanced_accuracy': val_stat.get('balanced_accuracy', val_stat['score']),
            'ood_test_balanced_accuracy': test_stat.get('balanced_accuracy', test_stat['score']),
            'id_val_balanced_accuracy': id_val_stat.get('balanced_accuracy', id_val_stat['score']),
            'id_test_balanced_accuracy': id_test_stat.get('balanced_accuracy', id_test_stat['score']),
            'val_score': val_stat['score'],
            'val_loss': val_stat['loss'],
            'test_score': test_stat['score'],
            'test_loss': test_stat['loss'],
            'id_val_subject_num': id_val_stat['subject_num'],
            'id_test_subject_num': id_test_stat['subject_num'],
            'ood_val_subject_num': val_stat['subject_num'],
            'ood_test_subject_num': test_stat['subject_num'],
            'epoch': epoch,
            'max epoch': config.train.max_epoch
        }

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
            'ood_val_balanced_accuracy': val_stat.get('balanced_accuracy', val_stat['score']),
            'ood_test_balanced_accuracy': test_stat.get('balanced_accuracy', test_stat['score']),
            'id_val_balanced_accuracy': id_val_stat.get('balanced_accuracy', id_val_stat['score']),
            'id_test_balanced_accuracy': id_test_stat.get('balanced_accuracy', id_test_stat['score']),
            'val_score': val_stat['score'],
            'val_loss': val_stat['loss'],
            'test_score': test_stat['score'],
            'test_loss': test_stat['loss'],
            'mixed_val_score': (val_stat['score'] * val_stat['subject_num'] + id_val_stat['score'] * id_val_stat['subject_num']) / (
                    val_stat['subject_num'] + id_val_stat['subject_num']),
            'mixed_test_score': (test_stat['score'] * test_stat['subject_num'] + id_test_stat['score'] * id_test_stat['subject_num']) / (
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
