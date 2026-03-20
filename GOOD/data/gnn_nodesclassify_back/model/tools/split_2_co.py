import numpy as np
from .find_major import find_major
from .find_major_num import find_major_num
import networkx as nx


def split_2_co(graph, id_dict, Ci, index, total_degree_dict):
    data = Ci[0]  # 数据节点位置信息
    #FL_matrix = Ci[1]  # Floyd矩阵信息
    degree_dict = Ci[1]  # 度字典
    #print(degree_dict,len(degree_dict))
    ball_1 = []  # 存储属于index[0]的点索引
    ball_2 = []  # 存储属于index[1]的点索引
    #print("index[0]", index[0])
    #print("index[1]", index[1])
    # 进行分裂
    # for i in degree_dict:
    #     if total_Fl[index[0]][i] < total_Fl[index[1]][i]:
    #         ball_1.append(i)
    #     else:
    #         ball_2.append(i)
    #存储子图索引的列表
    subnodes = []
    #sub=[]
    for node in data:
        subnodes.append(node[0])
    # for node in degree_dict:
    #     sub.append(node)
    #构建子图
    subgraph = graph.subgraph(subnodes)
    #subgraph2 = graph.subgraph(sub)
    #print("sub",subgraph.nodes(),len(subgraph.nodes()))
    #print("sub", subgraph2.nodes(), len(subgraph2.nodes()))
    # Dij_1 = nx.single_source_dijkstra_path_length(subgraph, id_dict[index[0]], weight="weight")
    # Dij_2 = nx.single_source_dijkstra_path_length(subgraph, id_dict[index[1]], weight="weight")
    # #print("1",Dij_1,len(Dij_1))
    # #print("2",Dij_2,len(Dij_2))
    # for new_id in degree_dict:
    #     if Dij_1[id_dict[new_id]] < Dij_2[id_dict[new_id]]:
    #         ball_1.append(new_id)
    #     else:
    #         ball_2.append(new_id)

    Distance_1 = nx.single_source_shortest_path_length(subgraph, id_dict[index[0]])
    Distance_2 = nx.single_source_shortest_path_length(subgraph, id_dict[index[1]])

    for new_id in degree_dict:
        if Distance_1[id_dict[new_id]] <= Distance_2[id_dict[new_id]]:
            ball_1.append(new_id)
        else:
            ball_2.append(new_id)

    #print("data：", len(data))
    #print("ball_1：", len(ball_1))
    #print("ball_2：", len(ball_2))
    ball_1 = np.array(ball_1).astype(int)
    ball_2 = np.array(ball_2).astype(int)
    #print("ball1:",ball_1)
    #print("ball2:", ball_2)
    # 对点信息进行分裂
    #data1 = data[ball_1]
    #data2 = data[ball_2]
    data1 = []
    data2 = []
    for i in ball_1:
        for j in data:
            if j[-1] == i:
                data1.append(j)
    for i in ball_2:
        for j in data:
            if j[-1] == i:
                data2.append(j)

    # 对floyd矩阵进行分裂
    # FL_matrix1 = total_Fl[np.ix_(ball_1, ball_1)]
    # FL_matrix2 = total_Fl[np.ix_(ball_2, ball_2)]

    # 对度字典分裂
    d1 = {}
    d2 = {}
    for i in range(len(ball_1)):
        d1.update({ball_1[i]: total_degree_dict[ball_1[i]]})

    for i in range(len(ball_2)):
        d2.update({ball_2[i]: total_degree_dict[ball_2[i]]})

    #print("du:",d1,d2)

    #计算粒球的纯度
    major_label1 = find_major(data1)  # 最多的标签
    major_label_num1, num_len1 = find_major_num(data1, major_label1)  # 最多的标签的点个数

    major_label2 = find_major(data2)  # 最多的标签
    major_label_num2, num_len2 = find_major_num(data2, major_label2)  # 最多的标签的点个数

    # 将分裂的点信息和边信息组合成两个球结构
    if num_len1 == 0:
        C1 = []
    else:
        C1 = [data1, d1, float(major_label_num1/num_len1)]

    if num_len2 == 0:
        C2 = []
    else:
        C2 = [data2, d2, float(major_label_num2/num_len2)]


    return C1, C2
