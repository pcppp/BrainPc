import time

import numpy as np
from math import sqrt
ini = float('inf')
from operator import itemgetter
#from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import networkx as nx
def initial_splite(C, graph, id_dict, id_dict_oldtonew, labels, total_degree_dict):

    connected_components = list(nx.connected_components(graph))
    print("连通数",len(connected_components))
    new_clusters = [] #存储每个连通分支的粒球
    for i, component in enumerate(connected_components, start=1):
        #print("连通", i ,list(component))
        subgraph = graph.subgraph(list(component))
        new_node_ids = []
        for old_id in list(component):
            new_node_ids.append(id_dict_oldtonew[old_id])

        component_data = [C[0][0][i] for i in new_node_ids] #提取连通分支的数据
        component_degree_dict = {node: total_degree_dict[node] for node in new_node_ids} #提取连通分支的度序列
        component_C = [component_data, component_degree_dict]
        centers = select_initial_centers(component_C, component_degree_dict) #选取中心点
        #print(centers)
        # Dijs = [] #存储每个中心点的迪杰斯特拉
        # for center in centers:
        #     Dij = nx.single_source_dijkstra_path_length(subgraph, id_dict[center], weight="weight")
        #     Dijs.append(Dij)
        #print("DIJS", Dijs)
        if len(centers) != 0:
            Distances = [] #存储每个中心点距离其他点的跳数
            for center in centers:
                Distance = nx.single_source_shortest_path_length(subgraph, id_dict[center])
                Distances.append(Distance)

            #初始化记录粒球的列表
            balls_data_nodes = [[] for _ in centers]
            balls_degree_dict = [{} for _ in centers]
            for old_id in list(component): #找到每个点距离最近的中心点索引
                min = np.inf
                min_center_idx = -1
                # for Dij in Dijs: #遍历每个中心点的迪杰斯特拉
                #     if Dij[old_id] < min:
                #         min = Dij[old_id]
                #         min_center_idx = next(iter(Dij.keys()))
                for Distance in Distances: #遍历每个中心点可达的跳数字典
                    if Distance[old_id] < min:
                        min = Distance[old_id]
                        min_center_idx = next(iter(Distance.keys())) #更新所遍历点跳数最少的中心点

                #将点归入最近中心点的粒球
                #print("min", min_center_idx)
                center_index = centers.index(id_dict_oldtonew[min_center_idx])
                balls_data_nodes[center_index].append(C[0][0][id_dict_oldtonew[old_id]])

            # 更新每个球的度字典
            for index, center in enumerate(centers):
                ball_nodes = balls_data_nodes[index]
                # [row[-2] for row in component_data]
                ball_nodes_index = [int(row[-1]) for row in ball_nodes]

                for node in ball_nodes_index:
                    balls_degree_dict[index][node] = total_degree_dict[node]

                    # 组合每个球的数据结构
            balls = [
                [balls_data_nodes[i], balls_degree_dict[i]]
                for i in range(len(centers))
            ]
            for ball in balls:
                new_clusters.append(ball)
    return new_clusters


def select_initial_centers(component_C, degree_dict):
    # component_C[0] 包含了所有的节点信息，其中每个元素的最后第二列是标签，最后一列是索引
    node_info = component_C[0]

    # 初始化存储每个类别的节点索引
    class_nodes_dict = {}

    # 遍历所有节点，根据标签组织节点索引
    for info in node_info:
        label = info[-2]  # 获取标签
        node_index = int(info[-1])  # 获取索引
        if label not in class_nodes_dict and label != -1:
            class_nodes_dict[label] = []
        if label != -1:
            class_nodes_dict[label].append(node_index)

    # 计算类别数
    num_classes = len(class_nodes_dict)
    #print("num_class",num_classes)
    # 数据集中的节点数
    num_nodes = len(node_info)

    # 计算每个类别应选取的中心点数
    if num_classes != 0:
        centers_per_class = max(1, int(sqrt(num_nodes) / num_classes))  #可改为根号n/2

    # 选取中心点
    centers = []
    for label, nodes in class_nodes_dict.items():
        # 获取这些节点的度，并按度从高到低排序
        degrees = [(node, degree_dict[node]) for node in nodes if node in degree_dict]
        degrees.sort(key=lambda x: x[1], reverse=True)

        # 选取前k个节点作为中心点
        selected_centers = [node for node, _ in degrees[:centers_per_class]]
        centers.extend(selected_centers)
    #print("centers", centers)
    return centers



