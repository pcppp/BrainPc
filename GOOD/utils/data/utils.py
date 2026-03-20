import graph_tool as gt
import graph_tool.topology as top
import numpy as np
import torch
import gudhi as gd
import itertools
import networkx as nx

from tqdm import tqdm
from GOOD.utils.data.complex import Cochain, Complex
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from GOOD.utils.data.parallel import ProgressParallel
from joblib import delayed

def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    """Constructs a simplex tree from a PyG graph.

    Args:
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph.
    """
    st = gd.SimplexTree()
    # Add vertices to the simplex.
    for v in range(size):
        st.insert([v])

    # Add the edges to the simplex.
    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st


def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]


def build_tables(simplex_tree, size):
    complex_dim = simplex_tree.dimension()
    # Each of these data structures has a separate entry per dimension.
    id_maps = [{} for _ in range(complex_dim+1)] # simplex -> id
    simplex_tables = [[] for _ in range(complex_dim+1)] # matrix of simplices
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue

        # Assign this simplex the next unused ID
        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps


def extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim: int):
    """Build two maps simplex -> its coboundaries and simplex -> its boundaries"""
    # The extra dimension is added just for convenience to avoid treating it as a special case.
    boundaries = [{} for _ in range(complex_dim+2)]  # simplex -> boundaries
    coboundaries = [{} for _ in range(complex_dim+2)]  # simplex -> coboundaries
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    for simplex, _ in simplex_tree.get_simplices():
        # Extract the relevant boundary and coboundary maps
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        # Add the boundaries of the simplex to the boundaries table
        if simplex_dim > 0:
            boundaries_ids = [id_maps[simplex_dim-1][boundary] for boundary in get_simplex_boundaries(simplex)]
            boundaries_tables[simplex_dim].append(boundaries_ids)

        # This operation should be roughly be O(dim_complex), so that is very efficient for us.
        # For details see pages 6-7 https://hal.inria.fr/hal-00707901v1/document
        simplex_coboundaries = simplex_tree.get_cofaces(simplex, codimension=1)
        for coboundary, _ in simplex_coboundaries:
            assert len(coboundary) == len(simplex) + 1

            if tuple(simplex) not in level_coboundaries:
                level_coboundaries[tuple(simplex)] = list()
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))

            if tuple(coboundary) not in level_boundaries:
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    return boundaries_tables, boundaries, coboundaries


def build_adj(boundaries: List[Dict], coboundaries: List[Dict], id_maps: List[Dict], complex_dim: int,
              include_down_adj: bool):
    """Builds the upper and lower adjacency data structures of the complex

    Args:
        boundaries: A list of dictionaries of the form
            boundaries[dim][simplex] -> List[simplex] (the boundaries)
        coboundaries: A list of dictionaries of the form
            coboundaries[dim][simplex] -> List[simplex] (the coboundaries)
        id_maps: A dictionary from simplex -> simplex_id
    """
    def initialise_structure():
        return [[] for _ in range(complex_dim+1)]

    upper_indexes, lower_indexes = initialise_structure(), initialise_structure()
    all_shared_boundaries, all_shared_coboundaries = initialise_structure(), initialise_structure()

    # Go through all dimensions of the complex
    for dim in range(complex_dim+1):
        # Go through all the simplices at that dimension
        for simplex, id in id_maps[dim].items():
            # Add the upper adjacent neighbours from the level below
            if dim > 0:
                for boundary1, boundary2 in itertools.combinations(boundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim - 1][boundary1], id_maps[dim - 1][boundary2]
                    upper_indexes[dim - 1].extend([[id1, id2], [id2, id1]])
                    all_shared_coboundaries[dim - 1].extend([id, id])

            # Add the lower adjacent neighbours from the level above
            if include_down_adj and dim < complex_dim and simplex in coboundaries[dim]:
                for coboundary1, coboundary2 in itertools.combinations(coboundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim + 1][coboundary1], id_maps[dim + 1][coboundary2]
                    lower_indexes[dim + 1].extend([[id1, id2], [id2, id1]])
                    all_shared_boundaries[dim + 1].extend([id, id])

    return all_shared_boundaries, all_shared_coboundaries, lower_indexes, upper_indexes


def construct_features(vx: Tensor, cell_tables, init_method: str) -> List:
    """Combines the features of the component vertices to initialise the cell features"""
    features = [vx]
    for dim in range(1, len(cell_tables)):
        aux_1 = []
        aux_0 = []
        for c, cell in enumerate(cell_tables[dim]):
            aux_1 += [c for _ in range(len(cell))]
            aux_0 += cell
        node_cell_index = torch.LongTensor([aux_0, aux_1])
        in_features = vx.index_select(0, node_cell_index[0])
        features.append(scatter(in_features, node_cell_index[1], dim=0,
                                dim_size=len(cell_tables[dim]), reduce=init_method))

    return features


def extract_labels(y, size):
    v_y, complex_y = None, None
    if y is None:
        return v_y, complex_y

    # ---------------------------------------------ADD-------------------------------------------------------
    # 处理不同类型的y
    if isinstance(y, (int, float)):
        # 如果y是标量，将其转换为张量
        complex_y = torch.tensor([y])
        return v_y, complex_y
    
    if isinstance(y, np.ndarray):
        # 如果y是numpy数组，转换为张量
        y = torch.from_numpy(y)
    
    if not isinstance(y, torch.Tensor):
        # 如果y是其他类型，尝试转换为张量
        try:
            y = torch.tensor(y)
        except Exception as e:
            print(f"警告: 无法转换标签类型 {type(y)}: {e}")
            return v_y, complex_y

    # 确保y是张量
    if not hasattr(y, 'size'):
        print(f"警告: 标签对象没有size方法: {type(y)}")
        return v_y, complex_y

    # 处理0维张量（标量张量）
    if y.dim() == 0:
        complex_y = y.unsqueeze(0)  # 转换为1维张量
        return v_y, complex_y
    #--------------------------------------------------------------------------------------------------------


    y_shape = list(y.size())

    if y_shape[0] == 1:
        # This is a label for the whole graph (for graph classification).
        # We will use it for the complex.
        complex_y = y
    else:
        # This is a label for the vertices of the complex.
        assert y_shape[0] == size
        v_y = y

    return v_y, complex_y


def generate_cochain(dim, x, all_upper_index, all_lower_index,
                   all_shared_boundaries, all_shared_coboundaries, cell_tables, boundaries_tables,
                   complex_dim, y=None):
    """Builds a Cochain given all the adjacency data extracted from the complex."""
    if dim == 0:
        assert len(all_lower_index[dim]) == 0
        assert len(all_shared_boundaries[dim]) == 0

    num_cells_down = len(cell_tables[dim-1]) if dim > 0 else None
    num_cells_up = len(cell_tables[dim+1]) if dim < complex_dim else 0

    up_index = (torch.tensor(all_upper_index[dim], dtype=torch.long).t()
                if len(all_upper_index[dim]) > 0 else None)
    down_index = (torch.tensor(all_lower_index[dim], dtype=torch.long).t()
                  if len(all_lower_index[dim]) > 0 else None)
    shared_coboundaries = (torch.tensor(all_shared_coboundaries[dim], dtype=torch.long)
                      if len(all_shared_coboundaries[dim]) > 0 else None)
    shared_boundaries = (torch.tensor(all_shared_boundaries[dim], dtype=torch.long)
                    if len(all_shared_boundaries[dim]) > 0 else None)
    
    boundary_index = None
    if len(boundaries_tables[dim]) > 0:
        boundary_index = [list(), list()]
        for s, cell in enumerate(boundaries_tables[dim]):
            for boundary in cell:
                boundary_index[1].append(s)
                boundary_index[0].append(boundary)
        boundary_index = torch.LongTensor(boundary_index)
        
    if num_cells_down is None:
        assert shared_boundaries is None
    if num_cells_up == 0:
        assert shared_coboundaries is None

    if up_index is not None:
        assert up_index.size(1) == shared_coboundaries.size(0)
        assert num_cells_up == shared_coboundaries.max() + 1
    if down_index is not None:
        assert down_index.size(1) == shared_boundaries.size(0)
        assert num_cells_down >= shared_boundaries.max() + 1

    return Cochain(dim=dim, x=x, upper_index=up_index,
                 lower_index=down_index, shared_coboundaries=shared_coboundaries,
                 shared_boundaries=shared_boundaries, y=y, num_cells_down=num_cells_down,
                 num_cells_up=num_cells_up, boundary_index=boundary_index)


def compute_clique_complex_with_gudhi(x: Tensor, edge_index: Adj, size: int,
                                      expansion_dim: int = 2, y: Tensor = None,
                                      include_down_adj=True,
                                      init_method: str = 'sum') -> Complex:
    """Generates a clique complex of a pyG graph via gudhi.

    Args:
        x: The feature matrix for the nodes of the graph
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph
        expansion_dim: The dimension to expand the simplex to.
        y: Labels for the graph nodes or a label for the whole graph.
        include_down_adj: Whether to add down adj in the complex or not
        init_method: How to initialise features at higher levels.
    """
    assert x is not None
    assert isinstance(edge_index, Tensor)  # Support only tensor edge_index for now

    # Creates the gudhi-based simplicial complex
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    simplex_tree.expansion(expansion_dim)  # Computes the clique complex up to the desired dim.
    complex_dim = simplex_tree.dimension()  # See what is the dimension of the complex now.

    # Builds tables of the simplicial complexes at each level and their IDs
    simplex_tables, id_maps = build_tables(simplex_tree, size)

    # Extracts the boundaries and coboundaries of each simplex in the complex
    boundaries_tables, boundaries, co_boundaries = (
        extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim))

    # Computes the adjacencies between all the simplexes in the complex
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                   complex_dim, include_down_adj)

    # Construct features for the higher dimensions
    # TODO: Make this handle edge features as well and add alternative options to compute this.
    xs = construct_features(x, simplex_tables, init_method)

    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    cochains = []
    for i in range(complex_dim+1):
        y = v_y if i == 0 else None
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                               simplex_tables, boundaries_tables, complex_dim=complex_dim, y=y)
        cochains.append(cochain)

    return Complex(*cochains, y=complex_y, dimension=complex_dim)

def remove_self_loops_and_parallel_edges(graph_gt):
    """替代gt.stats方法的实现"""
    
    # 1. 移除自环 - 使用graph-tool的其他方法
    try:
        # 方法1: 使用remove_edge删除自环
        vertices_to_check = list(graph_gt.vertices())
        for v in vertices_to_check:
            # 检查是否有自环
            if graph_gt.edge(v, v) is not None:
                graph_gt.remove_edge(graph_gt.edge(v, v))
    except:
        # 方法2: 重新构建图来移除自环
        edges_without_self_loops = []
        for e in graph_gt.edges():
            if e.source() != e.target():
                edges_without_self_loops.append((int(e.source()), int(e.target())))
        
        # 重新创建图
        num_vertices = graph_gt.num_vertices()
        graph_gt.clear()
        graph_gt.add_vertex(num_vertices)
        if edges_without_self_loops:
            graph_gt.add_edge_list(edges_without_self_loops)
    
    # 2. 移除平行边 - graph-tool默认不允许平行边，所以通常不需要处理
    # 但如果需要，可以通过重新构建图来确保没有重复边
    try:
        # 收集所有边，使用set去重
        edges = set()
        for e in graph_gt.edges():
            source, target = int(e.source()), int(e.target())
            # 对无向图，确保边的顺序一致
            edge = tuple(sorted([source, target]))
            edges.add(edge)
        
        # 如果边数发生变化，说明有平行边，需要重建
        if len(edges) != graph_gt.num_edges():
            num_vertices = graph_gt.num_vertices()
            graph_gt.clear()
            graph_gt.add_vertex(num_vertices)
            if edges:
                graph_gt.add_edge_list(list(edges))
    except:
        pass  # 如果失败，继续使用原图
def convert_graph_dataset_with_gudhi(dataset, expansion_dim: int, include_down_adj=True,
                                     init_method: str = 'sum'):
    # TODO(Cris): Add parallelism to this code like in the cell complex conversion code.
    dimension = -1
    complexes = []
    num_features = [None for _ in range(expansion_dim+1)]

    for data in tqdm(dataset):
        complex = compute_clique_complex_with_gudhi(data.x, data.edge_index, data.num_nodes,
            expansion_dim=expansion_dim, y=data.y, include_down_adj=include_down_adj,
            init_method=init_method)
        if complex.dimension > dimension:
            dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features
            else:
                assert num_features[dim] == complex.cochains[dim].num_features
        complexes.append(complex)

    return complexes, dimension, num_features[:dimension+1]


# ---- support for rings as cells

def get_rings(edge_index, max_k=7):
    print(f"        [环检测] 开始查找长度3-{max_k}的环...")
    
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    remove_self_loops_and_parallel_edges(graph_gt)
    # gt.stats.remove_self_loops(graph_gt)
    # gt.stats.remove_parallel_edges(graph_gt)
    
    print(f"        - 图统计: {graph_gt.num_vertices()} 个节点, {graph_gt.num_edges()} 条边")
    
    rings = set()
    sorted_rings = set()
    
    for k in range(3, max_k+1):
        print(f"        - 查找长度为 {k} 的环...")
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        
        count_k = 0
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
                count_k += 1
        
        print(f"          - 找到 {count_k} 个长度为 {k} 的环")
    
    rings = list(rings)
    print(f"        - 总共找到 {len(rings)} 个唯一环")
    return rings


def build_tables_with_rings(edge_index, simplex_tree, size, max_k):
    print("    [子步骤] 构建包含环的表格...")
    
    # 构建基础单纯表格
    cell_tables, id_maps = build_tables(simplex_tree, size)
    print(f"      - 基础单纯表格: {len(cell_tables)} 个维度")
    
    # 查找环
    print(f"      - 开始查找环 (最大长度: {max_k})...")
    rings = get_rings(edge_index, max_k=max_k)
    print(f"      - 找到 {len(rings)} 个环")
    
    if len(rings) > 0:
        # 添加环作为2-细胞
        id_maps += [{}]
        cell_tables += [[]]
        assert len(cell_tables) == 3, cell_tables
        
        for cell in rings:
            next_id = len(cell_tables[2])
            id_maps[2][cell] = next_id
            cell_tables[2].append(list(cell))
        
        print(f"      - 添加环作为2-细胞: {len(cell_tables[2])} 个环")

    return cell_tables, id_maps
def get_ring_boundaries(ring):
    boundaries = list()
    for n in range(len(ring)):
        a = n
        if n + 1 == len(ring):
            b = 0
        else:
            b = n + 1
        # We represent the boundaries in lexicographic order
        # so to be compatible with 0- and 1- dim cells
        # extracted as simplices with gudhi
        boundaries.append(tuple(sorted([ring[a], ring[b]])))
    return sorted(boundaries)


def extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps):
    """Build two maps: cell -> its coboundaries and cell -> its boundaries"""

    # Find boundaries and coboundaries up to edges by conveniently
    # invoking the code for simplicial complexes
    assert simplex_tree.dimension() <= 1
    boundaries_tables, boundaries, coboundaries = extract_boundaries_and_coboundaries_from_simplex_tree(
                                            simplex_tree, id_maps, simplex_tree.dimension())
    
    assert len(id_maps) <= 3
    if len(id_maps) == 3:
        # Extend tables with boundary and coboundary information of rings
        boundaries += [{}]
        coboundaries += [{}]
        boundaries_tables += [[]]
        for cell in id_maps[2]:
            cell_boundaries = get_ring_boundaries(cell)
            boundaries[2][cell] = list()
            boundaries_tables[2].append([])
            for boundary in cell_boundaries:
                assert boundary in id_maps[1], boundary
                boundaries[2][cell].append(boundary)
                if boundary not in coboundaries[1]:
                    coboundaries[1][boundary] = list()
                coboundaries[1][boundary].append(cell)
                boundaries_tables[2][-1].append(id_maps[1][boundary])
    
    return boundaries_tables, boundaries, coboundaries

def compute_ring_2complex(x: Union[Tensor, np.ndarray], edge_index: Union[Tensor, np.ndarray],
                          edge_attr: Optional[Union[Tensor, np.ndarray]],
                          size: int, y: Optional[Union[Tensor, np.ndarray]] = None, max_k: int = 7,
                          include_down_adj=True, init_method: str = 'sum',
                          init_edges=True, init_rings=False) -> Complex:
    """Generates a ring 2-complex of a pyG graph via graph-tool."""
    
    print(f"[STEP 1] 开始构建环复形 - 节点数: {size}, 最大环长: {max_k}")
    assert x is not None
    assert isinstance(edge_index, np.ndarray) or isinstance(edge_index, Tensor)

    # 数据类型转换
    print("[STEP 2] 数据类型转换...")
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
        print(f"  - 节点特征转换为tensor: {x.shape}")
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.tensor(edge_index)
        print(f"  - 边索引转换为tensor: {edge_index.shape}")
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.tensor(edge_attr)
        print(f"  - 边特征转换为tensor: {edge_attr.shape if edge_attr is not None else None}")

    # 创建单纯树
    print("[STEP 3] 创建单纯树...")
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    print(f"  - 单纯树维度: {simplex_tree.dimension()}")
    print(f"  - 单纯树单纯形数量: {simplex_tree.num_simplices()}")
    
    assert simplex_tree.dimension() <= 1
    if simplex_tree.dimension() == 0:
        assert edge_index.size(1) == 0
        print("  - 警告: 图没有边，只有节点")

    # 构建包含环的表格
    print("[STEP 4] 构建细胞表格和ID映射...")
    cell_tables, id_maps = build_tables_with_rings(edge_index, simplex_tree, size, max_k)
    assert len(id_maps) <= 3
    complex_dim = len(id_maps)-1
    
    print(f"  - 复形维度: {complex_dim}")
    for dim in range(len(cell_tables)):
        if dim == 0:
            print(f"  - 0-细胞(节点)数量: {len(cell_tables[dim])}")
        elif dim == 1:
            print(f"  - 1-细胞(边)数量: {len(cell_tables[dim])}")
        elif dim == 2:
            print(f"  - 2-细胞(环)数量: {len(cell_tables[dim])}")
            if len(cell_tables[dim]) > 0:
                ring_sizes = [len(ring) for ring in cell_tables[dim]]
                print(f"    - 环长度分布: {dict(zip(*np.unique(ring_sizes, return_counts=True)))}")

    # 提取边界和余边界关系
    print("[STEP 5] 提取边界和余边界关系...")
    boundaries_tables, boundaries, co_boundaries = extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps)
    
    # 打印边界信息
    for dim in range(len(boundaries_tables)):
        if len(boundaries_tables[dim]) > 0:
            print(f"  - {dim}-细胞边界信息: {len(boundaries_tables[dim])} 个细胞有边界")

    # 计算邻接关系
    print("[STEP 6] 计算细胞间邻接关系...")
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                   complex_dim, include_down_adj)
    
    # 打印邻接信息
    for dim in range(len(upper_idx)):
        if len(upper_idx[dim]) > 0:
            print(f"  - {dim}-细胞上邻接数量: {len(upper_idx[dim])}")
        if len(lower_idx[dim]) > 0:
            print(f"  - {dim}-细胞下邻接数量: {len(lower_idx[dim])}")

    # 构建高维特征
    print("[STEP 7] 构建细胞特征...")
    xs = [x, None, None]
    constructed_features = construct_features(x, cell_tables, init_method)
    print(f"  - 构建的特征维度数: {len(constructed_features)}")
    
    if simplex_tree.dimension() == 0:
        assert len(constructed_features) == 1
        print("  - 只有节点特征")
    
    if init_rings and len(constructed_features) > 2:
        xs[2] = constructed_features[2]
        print(f"  - 初始化环特征: {xs[2].shape}")
    
    if init_edges and simplex_tree.dimension() >= 1:
        if edge_attr is None:
            xs[1] = constructed_features[1]
            print(f"  - 从节点特征构建边特征: {xs[1].shape}")
        else:
            print("  - 使用提供的边特征...")
            # 处理边特征的逻辑...
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
                print(f"    - 边特征重塑: {edge_attr.shape}")
            
            # 构建边特征字典
            ex = dict()
            for e, edge in enumerate(edge_index.numpy().T):
                canon_edge = tuple(sorted(edge))
                edge_id = id_maps[1][canon_edge]
                edge_feats = edge_attr[e]
                if edge_id in ex:
                    assert torch.equal(ex[edge_id], edge_feats)
                else:
                    ex[edge_id] = edge_feats

            # 构建边特征矩阵
            max_id = max(ex.keys())
            edge_feats = []
            assert len(cell_tables[1]) == max_id + 1
            for id in range(max_id + 1):
                edge_feats.append(ex[id])
            xs[1] = torch.stack(edge_feats, dim=0)
            print(f"    - 最终边特征矩阵: {xs[1].shape}")

    # 初始化标签
    print("[STEP 8] 初始化标签...")
    v_y, complex_y = extract_labels(y, size)
    if v_y is not None:
        print(f"  - 节点标签: {v_y.shape}")
    if complex_y is not None:
        print(f"  - 复形标签: {complex_y.shape}")

    # 生成cochain
    print("[STEP 9] 生成cochain...")
    cochains = []
    for i in range(complex_dim + 1):
        y_dim = v_y if i == 0 else None
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                               cell_tables, boundaries_tables, complex_dim=complex_dim, y=y_dim)
        cochains.append(cochain)
        
        if xs[i] is not None:
            print(f"  - {i}-cochain: {xs[i].shape[0]} 个细胞, {xs[i].shape[1]} 个特征")
        else:
            print(f"  - {i}-cochain: 无特征")

    print(f"[STEP 10] 复形构建完成 - 维度: {complex_dim}")
    return Complex(*cochains, y=complex_y, dimension=complex_dim)

def convert_graph_dataset_with_rings(dataset, max_ring_size=7, include_down_adj=False,
                                     init_method: str = 'sum', init_edges=True, init_rings=False,
                                     n_jobs=1):
    print(f"[数据集转换] 开始转换 {len(dataset)} 个图为环复形...")
    print(f"参数设置: max_ring_size={max_ring_size}, n_jobs={n_jobs}, init_method={init_method}")
    
    dimension = -1
    num_features = [None, None, None]

    def maybe_convert_to_numpy(x):
        if isinstance(x, Tensor):
            return x.numpy()
        return x

    # 并行处理数据集
    print("开始并行处理...")
    parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=len(dataset))
    complexes = parallel(delayed(compute_ring_2complex)(
        maybe_convert_to_numpy(data.x), maybe_convert_to_numpy(data.edge_index),
        maybe_convert_to_numpy(data.edge_attr),
        data.num_nodes, y=maybe_convert_to_numpy(data.y), max_k=max_ring_size,
        include_down_adj=include_down_adj, init_method=init_method,
        init_edges=init_edges, init_rings=init_rings) for data in dataset)

    print("验证转换结果...")
    # 验证转换结果
    for c, complex in enumerate(complexes):
        # 处理维度和特征数量
        if complex.dimension > dimension:
            dimension = complex.dimension
            print(f"更新最大维度: {dimension}")
            
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features
                print(f"设置 {dim}-维特征数: {num_features[dim]}")
            else:
                assert num_features[dim] == complex.cochains[dim].num_features

        # 验证图的一致性
        graph = dataset[c]
        if complex.y is not None:
            assert torch.equal(complex.y, graph.y)
        assert torch.equal(complex.cochains[0].x, graph.x)
        if complex.dimension >= 1:
            assert complex.cochains[1].x.size(0) == (graph.edge_index.size(1) // 2)

    print(f"[数据集转换完成] 最终维度: {dimension}, 特征数: {num_features[:dimension+1]}")
    return complexes, dimension, num_features[:dimension+1]


