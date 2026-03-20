import networkx as nx
import numpy as np

def add_weight(graph):
    node_positions = nx.get_node_attributes(graph, "pos")

    # 遍历每条边，计算距离并添加权重属性
    for u, v in graph.edges():

        pos_u = node_positions[u]
        pos_v = node_positions[v]
        distance = np.linalg.norm(np.array(pos_u) - np.array(pos_v))
        graph[u][v]['weight'] = distance
    return graph

def add_weight2(graph, type):

    node_positions = nx.get_node_attributes(graph, "pos")

    node_weights = nx.get_node_attributes(graph, 'attributes')

    # 遍历每条边，计算距离并添加权重属性
    for u, v in graph.edges():
        # print(node_positions[u])
        # print(node_weights[u])
        weights_u = np.concatenate((np.array(node_positions[u]), np.array(node_weights[u])), 0)
        weights_v = np.concatenate((np.array(node_positions[v]), np.array(node_weights[v])), 0)
        distance = np.linalg.norm(np.array(weights_u) - np.array(weights_v))
        graph[u][v]['weight'] = distance
        #print("inner weight",graph[u][v]['weight'])
        if type == 2:
            graph[u][v]['edge_attr'][0] = distance
    return graph