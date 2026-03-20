"""
Base class for OOD algorithms
"""
from abc import ABC
from torch import Tensor
from torch_geometric.data import Batch
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from typing import Tuple
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
import torch


class BaseOODAlg(ABC):
    r"""
    Base class for OOD algorithms

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(BaseOODAlg, self).__init__()
        self.optimizer: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.model: torch.nn.Module = None


        self.mean_loss = None
        self.spec_loss = None
        self.stage = 0
        self.current_mode = 'finetune'
    def set_stage(self, mode: str, config: Union[CommonArgs, Munch]):
        r"""
        Switch training stage between 'pretrain' and 'finetune'.
        
        Args:
            mode (str): 'pretrain' or 'finetune'
            config: config dictionary
        """
        print(f"#IN# Switching OOD Algorithm stage to: [{mode.upper()}]")
        self.current_mode = mode
        
        # 1. 让模型内部切换结构 (例如: 冻结某些层，切换 Head)
        # 前提：你的 self.model 需要实现 set_mode 方法
        if hasattr(self.model, 'set_mode'):
            self.model.set_mode(mode)
        
        # 2. 确定当前阶段的学习率
        # 通常微调阶段(finetune)的学习率比预训练要小，或者在 config 里区分配置
        lr = config.train.pre_lr
        t_epochs = config.train.pre_epoch
        if mode == 'finetune' and hasattr(config.train, 'lr'):
            lr = config.train.lr
            t_epochs = config.train.max_epoch
        
        # 3. 【关键】重置优化器
        # 因为参数变了 (ProjectionHead vs ClassifierHead)，必须重新注册 parameters
        # filter(lambda p: p.requires_grad, ...) 确保只优化没被冻结的层
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr,
            weight_decay=config.train.weight_decay
        )
        if mode == 'finetune':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=config.train.mile_stones,
            gamma=0.1
        )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=t_epochs, 
                eta_min=1e-6
            )
        # # 4. 重置 Scheduler (因为 optimizer 换了，scheduler 也要跟着换)
        # 
    def stage_control(self, config):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:
        r"""
        Set input data format and preparations

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns:
            - data (Batch) - Processed input data.
            - targets (Tensor) - Processed input labels.
            - mask (Tensor) - Processed NAN masks for data formats.
            - node_norm (Tensor) - Processed node weights for normalization.

        """
        return data, targets, mask, node_norm

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        return model_output

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor, config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate prediction loss without any special OOD constrains

        Args:
            raw_pred (Tensor): model predictions (for pretrain: (z1, z2) tuple; for finetune: logits)
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func: Accuracy}
                                   })


        Returns (Tensor):
            loss tensor (same format for both pretrain and finetune modes)

        """
        if self.current_mode == 'pretrain':
            # 预训练模式：计算对比学习损失 (InfoNCE / NT-Xent)
            # raw_pred 应该是 (z1, z2) 元组，包含两个视图的投影表示
            temperature = getattr(config.ood, 'temperature', 0.01)
            # 传入标签以支持监督对比学习
            contrastive_loss = self.calculate_contrastive_loss(raw_pred, temperature, labels=targets)
            
            # 返回标量损失（与 finetune 模式在 loss_postprocess 中处理方式一致）
            return contrastive_loss

        # === 模式 B: 微调 / 正常训练 (分类) ===
        else:
            # 原有的逻辑保持不变
            loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
            loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss
            return loss
   
    def calculate_contrastive_loss(self, features, temperature: float = 0.05, labels=None):
        """
        InfoNCE / NT-Xent loss with optional supervised contrastive learning support.
        
        Args:
            features: Tuple of (z1, z2) from two augmented views
            temperature: Temperature parameter for softmax
            labels: Optional class labels for supervised contrastive learning
                   If provided, samples with same label are treated as positives
        """
        z1, z2 = features
        
        # 统一解包逻辑
        if isinstance(z1, (tuple, list)): z1 = z1[0]
        if isinstance(z2, (tuple, list)): z2 = z2[0]

        batch_size = z1.size(0)
        
        # Batch size保护
        if batch_size < 2:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        device = z1.device

        # L2 Normalization
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)

        # Concat -> [2N, D]
        representations = torch.cat([z1, z2], dim=0)
        
        # Similarity Matrix -> [2N, 2N]
        similarity_matrix = torch.matmul(representations, representations.T) / temperature

        # 移除自对比（对角线）
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity_matrix.masked_fill_(self_mask, -9e15)

        # ==============================================
        # 监督对比学习 (Supervised Contrastive Learning)
        # ==============================================
        if labels is not None:
            # 处理labels维度
            if labels.dim() > 1:
                labels = labels.view(-1)[:batch_size]
            
            # 扩展labels: [label_0, ..., label_N, label_0, ..., label_N]
            labels = labels.to(device)
            labels_expanded = torch.cat([labels, labels], dim=0)  # [2N]
            
            # 创建正样本mask: 相同label的样本都是正样本
            # [2N, 2N] mask where mask[i,j]=True if labels[i]==labels[j] and i!=j
            label_mask = labels_expanded.unsqueeze(0) == labels_expanded.unsqueeze(1)
            label_mask = label_mask & ~self_mask  # 排除自己
            
            # 计算监督对比损失
            # 对于每个样本，所有同类样本都是正样本
            num_positives = label_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [2N, 1]
            
            # 计算log-sum-exp
            log_prob = similarity_matrix - torch.logsumexp(similarity_matrix,dim=1, keepdim=True)
                
            # 只对正样本计算loss
            mean_log_prob_pos = (label_mask * log_prob).sum(dim=1) / num_positives.squeeze()
            loss = -mean_log_prob_pos.mean()
            
        # ==============================================
        # 无监督对比学习 (Self-Supervised)
        # ==============================================
        else:
            # 原有逻辑：对于样本i，只有它的另一个增强视图是正样本
            # 构造标签: [N, N+1, ..., 2N-1, 0, 1, ..., N-1]
            contrast_labels = torch.cat([
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(0, batch_size, device=device)
            ], dim=0)
            
            loss = torch.nn.functional.cross_entropy(similarity_matrix, contrast_labels)
        
        return loss
        
    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns (Tensor):
            processed loss

        """
        if self.current_mode == 'pretrain':
            # 预训练模式：loss 已经是标量，直接返回
            self.mean_loss = loss
            return self.mean_loss
        else:
            # 微调/分类模式：原有逻辑
            self.mean_loss = loss.sum() / mask.sum()
            return self.mean_loss

    def set_up(self, model: torch.nn.Module, config: Union[CommonArgs, Munch]):
        r"""
        Training setup of optimizer and scheduler

        Args:
            model (torch.nn.Module): model for setup
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.train.lr`, :obj:`config.metric`, :obj:`config.train.mile_stones`)

        Returns:
            None

        """
        self.model: torch.nn.Module = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr,
                                          weight_decay=config.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.train.mile_stones,
                                                              gamma=0.1)

    def backward(self, loss):
        r"""
        Gradient backward process and parameter update.

        Args:
            loss: target loss
        """
        loss.backward()
        self.optimizer.step()
