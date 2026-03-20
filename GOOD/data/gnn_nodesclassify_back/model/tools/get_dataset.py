import os
import csv
import numpy as np
import networkx as nx


import time

def get_dataset(data_points, adjacency_matrix,  data_labels, seed=None,):
    start = time.perf_counter()
    # 创建一个无向图对象
    graph = nx.Graph()

    # 添加节点到图中
    for node_id, point in enumerate(data_points):
        graph.add_node(node_id, attributes=point)  # 添加节点以及它的特征

    #添加节点标签在图中
    for node_id, label in enumerate(data_labels):
        graph.nodes[node_id]['label'] = label
    end = time.perf_counter()
    #print("添加节点和标签的时间：", end - start)
    start = time.perf_counter()
    # 添加边到图中
    adjacency_matrix = adjacency_matrix.tocsr()  # 转换为 CSR 格式
    #print("csr:", adjacency_matrix)
    rows, cols = adjacency_matrix.nonzero()
    #print("rows,cols", rows, cols)
    for i, j in zip(rows, cols):
        if i != j:  # 检查是否有自环
            graph.add_edge(i, j,edge_attr=np.zeros(3))
    end = time.perf_counter()
    #print("加边时间:", end-start)


    return graph



