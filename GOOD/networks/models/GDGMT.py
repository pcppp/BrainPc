r"""
Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism <https://arxiv.org/abs/2201.12987>`_.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected
from torch_sparse import transpose

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor, DGINFeatExtractor
from .GAT import GATFeatExtractor
from .GINvirtualnode import vGINFeatExtractor, DvGINFeatExtractor
from .GCNs import DGCNFeatExtractor

class SimpleCNN(nn.Module):
    """Simple 1D CNN encoder for temporal window features."""
    
    def __init__(self, in_channels, out_channels, num_layers=2):
        super(SimpleCNN, self).__init__()
        
        # 简单的两层 CNN
        if num_layers == 1:
            self.conv1 = nn.Conv1d(1, out_channels, kernel_size=20, stride=1,padding=2)# 步长1 窗口5
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = None
        else:
            hidden_channels = out_channels // 2
            self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output_dim = out_channels
    
    def forward(self, x):
        """
        Args:
            x: (N, T) - 节点时序特征
        Returns:
            (N, out_channels) - 编码后的特征
        """
        # (N, T) -> (N, 1, T)
        x = x.unsqueeze(1)
        
        # 第一层卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 第二层卷积（如果有）
        if self.conv2 is not None:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
        
        # Global pooling: (N, C, T) -> (N, C, 1) -> (N, C)
        x = self.pool(x).squeeze(-1)
        return x
@register.model_register
class GDGMT(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GDGMT, self).__init__(config)
        cnn_out_channels = getattr(config.model, "cnn_out_channels", config.model.cnn_out_channels)
        lstm_hidden_size = getattr(config.model, "lstm_hidden_size", config.model.lstm_hidden_size)
        cnn_num_layers = getattr(config.model, "cnn_num_layers", 2)
        # self.cnn = SimpleCNN(
        #     in_channels=1,
        #     out_channels=cnn_out_channels,
        #     num_layers=cnn_num_layers
        # )
        self.use_cnn = getattr(config.model, "use_cnn", config.model.use_cnn)
        if(self.use_cnn):
            self.cnn = nn.Conv1d(1, cnn_out_channels, kernel_size=5, padding=2)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_size, batch_first=True)
            self.dropout = nn.Dropout(p=0.7)
            # 修改 config 中的 dim_node 为 CNN 输出维度
            config.dataset.dim_node = cnn_out_channels
        # ----------------origin---------------使用GIN
        # self.gnn = DGINFeatExtractor(config)
        # self.extractor = ExtractorMLP(config) # 边特征提取器
        # ----------------origin---------------使用GAT
        self.gnn = GATFeatExtractor(config)
        self.classifier = Classifier(config)
        self.learn_edge_att = True
        self.sampling_method = config.ood.extra_param[0]
        self.sampling_rounds = config.ood.extra_param[3]
        self.config = config

        self.causal_adj = None
        self.diffusion_loss = 0.0
        self.entropy_loss = 0.0

    def forward(self, *args, **kwargs):
        r"""
        The GSAT model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')
        # batch_size = data.batch[-1].item() + 1

        # ------------------------origin-------------------------
        # emb, self.diffusion_loss = self.gnn(*args, without_readout=True, **kwargs)
        # self.entropy_loss = self.gnn.ent_loss
        # 提取重要的图结构信息(提取哪些边是重要的,提取哪些子图结构对预测有因果关系)
        # att_log_logits = self.extractor(emb, data.edge_index, data.batch)

        # self.causal_adj = generate_adjacency_matrices(data.edge_index, att_log_logits.squeeze(), batch_size).clone()
        # # 计算边注意力
        # att = self.sampling(att_log_logits, self.training)


        # if self.learn_edge_att:
        #     if is_undirected(data.edge_index):
        #         nodesize = data.x.shape[0]
        #         edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
        #     else:
        #         edge_att = att
        # else:
        #     edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
                # 如果启用了 CNN,先对时序特征进行编码
        if self.use_cnn:
            # ---------- 1 ------------
            # encoded_feat = self.cnn(data.x)  # (N, cnn_out_dim)
            # data.x = encoded_feat
            # # 禁用 GNN 的 input_norm，因为 CNN 中已有 BatchNorm
            # self.gnn.input_norm = None

            #----------- 2 ------------
            x = data.x
            x = x.unsqueeze(1)  # (batch_size, seq_length) -> (batch_size, 1, seq_length)
            x = self.cnn(x)  # (batch_size, cnn_out_channels, seq_length)
            x = self.pool(x)  # (batch_size, cnn_out_channels, seq_length/2)
            x = x.transpose(1, 2)  # (batch_size, seq_length/2, cnn_out_channels)

            # LSTM 部分：学习长时依赖
            lstm_out, (h_n, c_n) = self.lstm(x)
            x = lstm_out
            # 方法 A: 取最后一个时间步 (捕捉最终状态)
            node_features = lstm_out[:, -1, :]     
            # 方法 B (可选): 取所有时间步的平均 (捕捉全局时序信息)
            # node_features = torch.mean(lstm_out, dim=1)
            node_features = self.dropout(node_features)
        sampling_logits = []
        sampling_trials = self.sampling_rounds
        # # 多次采样取平均,通过边概率控制哪些结构重要
        while len(sampling_logits)<sampling_trials:
            if(self.use_cnn):
                x_out, diff_loss = self.gnn(node_features,*args, **kwargs)
            else:
                x_out, diff_loss = self.gnn(*args, **kwargs)
            # 消融边特征掩码
            # b = torch.bernoulli(edge_att)
            # cur_edge_att = (b - edge_att).detach() + edge_att  # straight-through estimator
            # set_masks(cur_edge_att, self)
            # logits = self.classifier(x)
            # clear_masks(self)
            sampling_logits.append(self.classifier(x_out))
        # 平均多个子图的预测
        # logits = self.classifier(x_out)
        logits = torch.stack(sampling_logits).mean(dim=0)

        # return logits, att, edge_att
        return logits, None, None

    def sampling(self, att_log_logits, training):
        if self.sampling_method =="normal":
            att = self.normal_sample(att_log_logits, temp=1.0, training=training)
        elif self.sampling_method =="bern":
            att = self.concrete_sample(att_log_logits, temp=1.0, training=training)
        # att = self.concrete_sample(att_log_logits, temp=1.0, training=training)
        # att = self.normal_sample(att_log_logits, temp=0.8, training=training)
        # att = self.gumbel_softmax_sample(att_log_logits, temp=1, training=training)
        # att = self.logistic_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def normal_sample(logits, temp, training):
        if training:
            random_noise = torch.randn_like(logits)
            att_normal = ((logits + random_noise * temp).sigmoid())
        else:
            att_normal = logits.sigmoid()
        return att_normal

    @staticmethod
    def gumbel_softmax_sample(logits, temp, training):
        random_noise = torch.empty_like(logits).uniform_(1e-10, 1 - 1e-10)
        gumbel_noise = -torch.log(-torch.log(random_noise))
        y = (logits + gumbel_noise) / temp
        return torch.softmax(y, dim=-1)

    @staticmethod
    def logistic_sample(logits, temp, training):
        if training:
            random_noise = torch.empty_like(logits).uniform_(1e-10, 1 - 1e-10)
            logistic_noise = torch.log(random_noise) - torch.log(1 - random_noise)
            att_logistic = ((logits + logistic_noise) / temp).sigmoid()
        else:
            att_logistic = logits.sigmoid()
        return att_logistic


@register.model_register
class GDGMTvGIN(GDGMT):
    r"""
    The GIN virtual node version of GSAT.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GDGMTvGIN, self).__init__(config)
        # self.gnn = DvGINFeatExtractor(config)


@register.model_register
class GDGMTGCN(GDGMT):
    r"""
    The GIN virtual node version of GSAT.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GDGMTGCN, self).__init__(config)
        self.gnn = DGCNFeatExtractor(config)


class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = config.ood.extra_param[0]  # learn_edge_att
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                if batch.shape[0] == 0:
                    inputs = inputs
                else:
                    inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._apply_sigmoid = False
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None


def generate_adjacency_matrices(edge_index, edge_weights, bz, num_nodes=100):
    """
    Generate an adjacency matrix from edge indices and edge weights.

    Parameters:
    - edge_index (torch.Tensor): A tensor of shape [2, num_edges] containing the indices of the edges.
    - edge_weights (torch.Tensor): A tensor of shape [num_edges] containing the weights of the edges.
    - num_nodes (int): The number of nodes in the graph.

    Returns:
    - adjacency_matrix (torch.Tensor): The adjacency matrix of shape [num_nodes, num_nodes].
    """
    adjacency_matrix = torch.zeros((bz * num_nodes, bz * num_nodes), dtype=edge_weights.dtype).to(edge_weights.device)
    adjacency_matrix[edge_index[0], edge_index[1]] = edge_weights

    # split the adjacency matrix into multiple adjacency matrices
    adjacency_matrices = [adjacency_matrix[i * num_nodes: (i + 1) * num_nodes, i * num_nodes: (i + 1) * num_nodes] for i in range(bz)]
    return torch.stack(adjacency_matrices).to(edge_weights.device)
