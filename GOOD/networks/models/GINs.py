r"""
The Graph Neural Network from the `"How Powerful are Graph Neural Networks?"
<https://arxiv.org/abs/1810.00826>`_ paper.
"""
from typing import Callable, Optional
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .MolEncoders import AtomEncoder, BondEncoder
from torch.nn import Identity
from .Diffusion import DenoiseModel


@register.model_register
class GIN(GNNBasic):
    r"""
    The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):

        super().__init__(config)
        self.feat_encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out


class GINFeatExtractor(GNNBasic):
    r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`)
    """
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GINFeatExtractor, self).__init__(config)
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            self.encoder = GINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = GINEncoder(config, **kwargs)
            self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, edge_attr, batch, batch_size, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, batch, batch_size, **kwargs)
        return out_readout


class DGINFeatExtractor(GNNBasic):
    r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`)
    """
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(DGINFeatExtractor, self).__init__(config)
        print("config.dataset 的内容如下：")
        for k, v in config.dataset.items():
            print(f"{k}: {v}")
        print("config.ood 的内容如下：")
        for k, v in config.ood.items():
            print(f"{k}: {v}")
        print("启用CWN：",config.model.use_cwn)    
        self.use_cwn = config.model.use_cwn == True
        self.lin_0 = torch.nn.Linear(100, 1)
        self.lin_1 = torch.nn.Linear(1, 1)
        self.lin_2 = torch.nn.Linear(100, 1)
        self.cwn_model = CWN(
            in_channels_0=config.dataset.in_channels_0,              # 例如节点初始维度
            in_channels_1=config.dataset.in_channels_1,              # 若无初始特征可设为0
            in_channels_2=config.dataset.in_channels_2,              # 若无初始特征可设为0
            hid_channels=config.model.dim_hidden,
            n_layers=config.model.model_layer)
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            raise NotImplementedError
        else:
            self.encoder = GINEncoder(config, **kwargs)
            self.hp_encoder = HPGINEncoder(config, **kwargs)
            # self.mask_emb = nn.Parameter(torch.randn(config.model.dim_hidden, config.model.dim_hidden), requires_grad=True)
            self.mask_emb = nn.Parameter(torch.randn(config.dataset.in_channels_0, config.dataset.in_channels_0), requires_grad=True)
            self.feat_dropout = nn.Dropout(config.ood.feature_dropout)
            # self.fuse = nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)
            # self.fuse = nn.Linear(config.model.dim_hidden, config.model.dim_hidden)
            self.edge_feat = False
            self.ent_loss = 0.0
        # self.diffusion = DenoiseModel(config.model.dim_hidden, config.model.dim_hidden)

    def forward(self, *args, **kwargs):
        r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        x, edge_index, batch, batch_size,x_0,x_1,x_2,adjacency_1,incidence_2,incidence_1_t= self.arguments_read(*args, **kwargs)
        
        # x = self.feat_dropout(x)
        kwargs.pop('batch_size', 'not found')
        # generate a learnable mask
        dim = x.shape[-1]
        bz = int(x.shape[0] / dim)
        single_mask = torch.mm(self.mask_emb, self.mask_emb.t()).sigmoid()
        single_mask = self.feat_dropout(single_mask)
        mask = single_mask.repeat(bz, 1).view(bz * dim, dim)
        x = x * mask
        loss = 0.0
        gin_graph = self.encoder(x, edge_index, batch, batch_size, **kwargs)
        kwargs_node = kwargs.copy()  # 复制原始 kwargs
        kwargs_node['without_readout'] = True  # 设置为获取节点级特征
        gin_node_features = self.encoder(x, edge_index, batch, batch_size,  **kwargs_node)
        out_readout = gin_graph
        # 🔧 将节点级特征按图分割
        gin_node_list = []
        start_idx = 0
        for i in range(batch_size):
            # 提取当前图的节点特征
            num_nodes = 100
            graph_nodes = gin_node_features[start_idx:start_idx + num_nodes]
            gin_node_list.append(graph_nodes)
            start_idx += num_nodes
            
            # print(f"图 {i}: 节点数 = {num_nodes}, 节点特征形状 = {graph_nodes.shape}")
        if(self.use_cwn):
            cwn0_list = []  
            # 🔧 使用原始batch信息，避免信息损失
            from torch_geometric.utils import to_dense_batch
            for x0,x1,x2,adjacency1,incidence2,incidence1_t in zip(gin_node_list,x_1,x_2, adjacency_1, incidence_2, incidence_1_t):
                # x0 = self.lin_0(x0)
                # x1 = self.lin_1(x1) 
                # x2 = self.lin_2(x2)
                cwn_0, cwn_1, cwn_2 = self.cwn_model(x0, x1, x2, adjacency1, incidence2, incidence1_t)
                cwn0_list.append(cwn_0)
            from torch_scatter import scatter_mean
            torch.nn.BCEWithLogitsLoss()
            device = cwn0_list[0].device
            lens = torch.tensor([t.size(0) for t in cwn0_list], device=device)  # (num_graphs,)
            batch_index = torch.repeat_interleave(
                torch.arange(len(cwn0_list), device=device), lens
            )  # 形状 (sum V_i,)
            cwn0_cat = torch.cat(cwn0_list, dim=0)    
            # 🔧 使用 to_dense_batch 将节点特征按图分组
            cwn_dense, cwn_mask = to_dense_batch(cwn0_cat, batch)
            # cwn_dense: [batch_size, max_nodes, hidden_dim] = [64, 100, hidden_dim]
            # cwn_mask: [batch_size, max_nodes] = [64, 100] - 指示哪些节点是真实的
            graph_0 = scatter_mean(cwn0_cat, batch_index, dim=0)
            if(kwargs.get('without_readout')):
                cwn_graph = cwn0_cat
            else :
                cwn_graph = []
                for i in range(batch_size):
                    # 获取第 i 个图的真实节点特征
                    real_nodes = cwn_dense[i][cwn_mask[i]]  # [num_real_nodes, hidden_dim]
                    graph_feat = real_nodes.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                    cwn_graph.append(graph_feat)
                cwn_graph = torch.cat(cwn_graph, dim=0)  # [64, hidden_dim]
            if gin_graph.size(-1) != cwn_graph.size(-1):
                raise RuntimeError(f"GIN与CWN隐藏维度不一致: {gin_graph.size(-1)} vs {cwn_graph.size(-1)}")
            out_readout = cwn_graph

        if kwargs.get('without_readout'):
            self.ent_loss = self.entropy_loss(single_mask)
            post_diffusion = self.hp_encoder(out_readout, edge_index, batch, batch_size, **kwargs)
            post_diffusion = post_diffusion.view(-1, dim, dim)
            inner_product = torch.bmm(post_diffusion, post_diffusion.transpose(1, 2))  # Shape: (bz, n, n)
            result_x = torch.tanh(inner_product).view(-1, dim)
            loss = self.mse_loss(result_x * mask, x)
            # loss = self.mse_loss(result_x, x)
        return out_readout, loss

    def bern_sample(self, logits, temp=1.0, training=True):
        if training:
            random_noise = torch.empty_like(logits).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((logits + random_noise) / temp).sigmoid()
        else:
            att_bern = (logits).sigmoid()
        return att_bern

    def mse_loss(self, x_reconstructed, x):
        criterion = nn.MSELoss()
        return criterion(x_reconstructed, x)

    def entropy_loss(self, x):
        entropy = (torch.distributions.Categorical(logits=x).entropy()).mean()
        assert not torch.isnan(entropy)
        return entropy



from GOOD.utils.cw.topomodelx.nn.cell.cwn import CWN
from torch_geometric.nn import global_mean_pool




class NetWork(GNNBasic):
    def __init__( 
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        hid_channels=16,
        num_classes=1,
        n_layers=2,
    ):
        super().__init__()
        self.base_model = CWN(
            in_channels_0,
            in_channels_1,
            in_channels_2,
            hid_channels=hid_channels,
            n_layers=n_layers,
        )
        self.lin_0 = torch.nn.Linear(hid_channels, num_classes)
        self.lin_1 = torch.nn.Linear(hid_channels, num_classes)
        self.lin_2 = torch.nn.Linear(hid_channels, num_classes)

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        adjacency_1,
        incidence_2,
        incidence_1_t,
    ):
        x_0, x_1, x_2 = self.base_model(
            x_0, x_1, x_2, adjacency_1, incidence_2, incidence_1_t
        )
        x_0 = self.lin_0(x_0)
        x_1 = self.lin_1(x_1)
        x_2 = self.lin_2(x_2)

        # Take the average of the 2D, 1D, and 0D cell features. If they are NaN, convert them to 0.
        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0

        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0

        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0

        # Return the sum of the averages
        return (
            two_dimensional_cells_mean
            + one_dimensional_cells_mean
            + zero_dimensional_cells_mean
        )




























class GINEncoder(BasicEncoder):
    r"""
    The GIN encoder for non-molecule data, using the :class:`~GINConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):

        super(GINEncoder, self).__init__(config, *args, **kwargs)
        num_layer = config.model.model_layer
        self.without_readout = kwargs.get('without_readout')

        # self.atom_encoder = AtomEncoder(config.model.dim_hidden)

        if kwargs.get('without_embed'):
            self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
        else:
            self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(config.dataset.dim_node, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

        self.convs = nn.ModuleList(
            [
                gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
                for _ in range(num_layer - 1)
            ]
        )

        # if kwargs.get('without_embed'):
        #     self.conv1 = gnn.GINConv(nn.Linear(config.model.dim_hidden, config.model.dim_hidden))
        # else:
        #     self.conv1 = gnn.GINConv(nn.Linear(config.dataset.dim_node, config.model.dim_hidden))
        #
        # self.convs = nn.ModuleList(
        #     [
        #         gnn.GINConv(nn.Linear(config.model.dim_hidden, config.model.dim_hidden))
        #         for _ in range(num_layer - 1)
        #     ]
        # )

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): batch size

        Returns (Tensor):
            node feature representations
        """

        post_conv = x + self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            hidden_x = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                hidden_x = relu(hidden_x)
            post_conv = post_conv + dropout(hidden_x)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout


class GINMolEncoder(BasicEncoder):
    r"""The GIN encoder for molecule data, using the :class:`~GINEConv` operator for message passing.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GINMolEncoder, self).__init__(config, **kwargs)
        self.without_readout = kwargs.get('without_readout')
        num_layer = config.model.model_layer
        if kwargs.get('without_embed'):
            self.atom_encoder = Identity()
        else:
            self.atom_encoder = AtomEncoder(config.model.dim_hidden)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                            nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

        self.convs = nn.ModuleList(
            [
                GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                       nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                       nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout


class GINEConv(gnn.MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if hasattr(self.nn[0], 'in_features'):
            in_channels = self.nn[0].in_features
        else:
            in_channels = self.nn[0].in_channels
        self.bone_encoder = BondEncoder(in_channels)
        # if edge_dim is not None:
        #     self.lin = Linear(edge_dim, in_channels)
        #     # self.lin = Linear(edge_dim, config.model.dim_hidden)
        # else:
        #     self.lin = None
        self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if self.bone_encoder:
            edge_attr = self.bone_encoder(edge_attr)
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class HPGINConv(gnn.GINConv):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        x_filtered = x[0] - out

        return self.nn(x_filtered)


class HPGINEncoder(GINEncoder):
    r"""
    The GIN encoder for non-molecule data, using the :class:`~GINConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):

        super(HPGINEncoder, self).__init__(config, *args, **kwargs)
        num_layer = config.model.model_layer
        self.without_readout = kwargs.get('without_readout')

        # self.atom_encoder = AtomEncoder(config.model.dim_hidden)

        if kwargs.get('without_embed'):
            self.conv1 = HPGINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
        else:
            self.conv1 = HPGINConv(nn.Sequential(nn.Linear(config.dataset.dim_node, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

        self.convs = nn.ModuleList(
            [
                HPGINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
                for _ in range(num_layer - 1)
            ]
        )

        # if kwargs.get('without_embed'):
        #     self.conv1 = HPGINConv(nn.Linear(config.model.dim_hidden, config.model.dim_hidden))
        # else:
        #     self.conv1 = HPGINConv(nn.Linear(config.dataset.dim_node, config.model.dim_hidden))
        #
        # self.convs = nn.ModuleList(
        #     [
        #         HPGINConv(nn.Linear(config.model.dim_hidden, config.model.dim_hidden))
        #         for _ in range(num_layer - 1)
        #     ]
        # )
