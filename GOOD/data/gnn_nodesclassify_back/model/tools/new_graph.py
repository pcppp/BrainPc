# from model.gnn_nodesclassify_back.tools.calculate_center_and_radius import calculate_center_and_radius
# from tools.add_weight import add_weight2
import numpy as np
import networkx as nx

import time


def new_graph4(granular_ball_list, graph):
    length = len(granular_ball_list)
    graph_new = nx.Graph()
    nodes = np.array(graph.nodes())

    # 首先，为每个粒球添加节点到新图
    for i in range(length):
        granular_ball = granular_ball_list[i]
        graph_new.add_node(i, label=granular_ball[-1])  # 最后一个元素是标签

    # 构建原图的节点到粒球索引的映射
    node_to_granular_ball_index = {}
    for i, granular_ball in enumerate(granular_ball_list):
        for node in granular_ball[0]:  # 第一个元素是节点列表
            node_index = int(node[-1])  # 节点存储格式允许直接获取其索引
            node_to_granular_ball_index[nodes[node_index]] = i

    # 遍历原图的每条边，根据边连接的节点确定是否在粒球间添加边
    for u, v in graph.edges():
        if u in node_to_granular_ball_index and v in node_to_granular_ball_index:
            u_index = node_to_granular_ball_index[u]
            v_index = node_to_granular_ball_index[v]
            if u_index != v_index:  # 避免自环,若两个index不同,说明分属于不同的粒球,则此时需要添加边
                graph_new.add_edge(u_index, v_index)  # 添加粒球之间的边
    # 打印所有边
    # for edge in graph_new.edges():
    #     print(edge)

    return graph_new
