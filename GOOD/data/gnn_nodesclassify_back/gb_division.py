import queue
import time
import os
import csv
from .model.tools import *
import numpy as np
import networkx as nx
#import matplotlib

from .model.tools.add_noisy import add_noisy

#matplotlib.use('TkAgg')
import scipy.sparse as sp

from .model.tools.split_ball_purity import split_ball_purity,split_ball_purity_by2,split_ball_purity_unrRcursion
from .model.tools.purification import purification
from .model.tools.add_weight import add_weight2
from .model.tools.add_id import add_id
from .model.tools.new_graph import new_graph4

from .model.tools.calculate_center_and_radius import calculate_center_and_radius

from .model.tools.corse_split import initial_splite
from .model.tools.add_purity import add_purity
from .model.tools.split_ball_purity import split_ball_further
# torch.set_printoptions(threshold=np.inf)


def gb_division(data, args):
    orgin_data = data
    seed = 0
    # 加躁
    if args.noisy == 1:
        data = add_noisy(data)

    ini = float('inf')
    C = []  # 初始化球簇链表
    # 球簇的数据结构包含点信息,floyd矩阵（用于分裂）,weight矩阵（用于算质量）

    # 节点数据位置
    data_file = data.x.numpy()
    #print("data_file", len(data_file))
    #限制分裂的粒球总数
    total_balls_num = int(args.ball_r * len(data_file))
    #print("total", total_balls_num)

    start = time.perf_counter()
    # 边数据
    adjacency_matrix = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                     shape=(data.y.shape[0], data.y.shape[0]),
                                     dtype=np.float32)
    end = time.perf_counter()
    #print("sp.coo_matrix方法时间：", end-start)

    # 隐藏测试集标签
    # for i in range(0, len(data.test_mask)):
    #     if data.test_mask[i]:
    #         data.y[i] = -1



    # 标签数据
    data_labels = data.y.numpy()
    # print('数据维度:', data.x.shape)
    # print(data)
    # print('数据维度:', data.x)
    # print('数据维度(numpy):', data_file)
    # print('真实标签', data.y.shape)
    # print('真实标签', data.y.numpy())
    # print("边数据：", adjacency_matrix)
    # print("边shape", data.edge_index.shape[1])



    # 图生成
    graph = get_dataset.get_dataset(data_file, adjacency_matrix, data_labels, seed)






    # 删除孤立点
    graph = del_outlier.del_outlier(graph)
    print("len(node)", len(graph.nodes()))

    # 记录粒球生成时间
    total_time = 0
    start_time1 = time.perf_counter()


    #print(data)
    # 获取点的属性（特征）
    node_attributes = nx.get_node_attributes(graph, "attributes")
    # print(node_attributes)
    attributes = np.array(list(node_attributes.values()))
    #print(attributes)
    # 获取点的标签
    node_labels = nx.get_node_attributes(graph, "label")
    labels = np.array(list(node_labels.values()))
    labels = np.expand_dims(labels, axis=0)
    labels = np.reshape(labels, (nx.number_of_nodes(graph), 1))
    #print(labels)

    #new_graph = get_dataset_2(attributes, adjacency_matrix, labels, seed) #绘制删除孤立点后的图

    #求节点的度
    total_degree_dict_old = dict(graph.degree())

    total_degree_dict = {}
    id = 0
    for key, value in total_degree_dict_old.items():
        total_degree_dict[id] = value
        id += 1
    #print("dict(old)", total_degree_dict_old)
    #print("len_dic(old)", len(total_degree_dict_old))
    #print("dict", total_degree_dict)
    #print("len_dic",len(total_degree_dict))

    #对应新老id的字典（新：老）
    id_dict = {}
    id_dict_oldtonew = {}
    for new, old in enumerate(total_degree_dict_old):
        id_dict[new] = old
        id_dict_oldtonew[old] = new
    #print("id_dic", id_dict)


    #加原始id
    indices = []
    for index in total_degree_dict_old:
        indices.append(index)
    #print("old id",indices)
    #print("graph id",graph.nodes())
    indices = np.expand_dims(indices, axis=0)
    indices = np.reshape(indices, (nx.number_of_nodes(graph), 1))
    #data = np.concatenate((indices, data, attributes, labels), axis=1)
    data = np.concatenate((indices, attributes, labels), axis=1)

    data = add_id(data)
    #print("data:", data)

    C.append([data, total_degree_dict])

    # 整个数据集二分裂
    # new_C = split_ball_purity_by2(C, labels, total_degree_dict, total_FL_matrix)

    end_time1 = time.perf_counter()
    #print("粗分前准备：", end_time1-start_time1, "秒")
    total_time += end_time1-start_time1
    start_time2 = time.perf_counter()
    # 先粗划分再二分裂
    new_C = initial_splite(C, graph, id_dict, id_dict_oldtonew, labels, total_degree_dict)
    #print("粗分后：", new_C)
    #若粗分球个数已经比阈值大则丢弃一部分粒球，解决citeseer无法降低粗化率问题
    target = 1
    while len(new_C) > total_balls_num:
        cut_pos = 0
        new_C.sort(key=lambda x: len(x[0]))
        for i in range(0, len(new_C)):
            # print("粗分球点数:", len(c[0]))
            if len(new_C[i][0]) == target+1:
                cut_pos = i
                break
        target += 1
        new_C = new_C[cut_pos:]
    # for c in new_C:
    #     print("每个球质量", len(c[0]))
    # print(len(new_C))


    # 给粒球添加纯度
    new_C = add_purity(new_C)
    # for C in new_C:
    #     print(C[-1])
    #存储粗分后的粒球的队列
    # C_queue = queue.Queue()
    # for item in new_C:
    #     C_queue.put(item)

    end_time2 = time.perf_counter()
    #print("粗分时间：", end_time2-start_time2, "秒")
    total_time += end_time2-start_time2
    start_time3 = time.perf_counter()
    # 球簇二分裂
    #new_C = split_ball_purity(graph, id_dict, C_queue, new_C, labels, total_degree_dict, total_balls_num)
    new_C = split_ball_purity_unrRcursion(graph, id_dict, new_C, total_degree_dict, total_balls_num)
    #如果当前粒球总数小于阈值，则继续分裂先分裂质量大的，解决无法达到0.5粗化率的问题
    if len(new_C) < total_balls_num:
        new_C = split_ball_further(graph, id_dict, new_C, total_degree_dict, total_balls_num)
    end_time3 = time.perf_counter()
    #print("二分裂时间：", end_time3-start_time3, "秒")
    total_time += end_time3-start_time3

    start_time4 = time.perf_counter()

    
    # for C in new_C:
    #     print("合并完粒球纯度：", C[-1])
    print("分完--------------------------------------------")
    #print("new C", new_C)
    #print("提纯前", len(new_C))
    # 提纯球簇并加标签
    #start = time.perf_counter()
    new_C = purification(new_C)
    #end = time.perf_counter()
    #print("提纯时间：", end-start)
    #print("提纯后", len(new_C))  #[[oldid,特征，标签，newid], 度序列, 纯度, 标签]
    print("粒球数", len(new_C))  #[[oldid,特征，标签，newid], 度序列, 纯度, 标签]


    #计算粒球特征平均
    GB_features = []
    for GB in new_C:
        data = np.array(GB[0])
        #print("data",data)
        #print("data(特征列)", data[:, 2:-2])
        slice = data[:, 1:-2]
        #print("slice", slice)
        feature = slice.mean(axis=0)
        GB_features.append(feature)

    # 优化加速
    # GB_graph = new_graph2(new_C, graph)
    GB_graph = new_graph4(new_C, graph)



    new_f = {}  # 初始化用于存储处理后数据的字典

    #x = []
    gb_labels = []
    for GB in new_C:
        gb_labels.append(GB[-1])  #顺序返回每个粒球的标签


    new_f['gb_labels'] = np.array(gb_labels)

    C_adj = sp.coo_matrix(nx.to_numpy_array(GB_graph))
    C_adj = np.vstack((C_adj.row, C_adj.col))
    # adj_matrix = np.array(nx.to_numpy_array(GB_graph))

    new_f['adj'] = C_adj



    new_f['gb_features'] = np.array(GB_features)


    def z_score_standardization(data):
        mean_vals = np.mean(data, axis=0)
        std_devs = np.std(data, axis=0)
        data = (data - mean_vals) / std_devs
        return data


    # 显示图的信息
    def display_graph_info(graph):
        print("图的信息：")
        print("节点数：", graph.number_of_nodes())
        print("边数：", graph.number_of_edges())
        print("平均度：", sum(dict(graph.degree()).values()) / graph.number_of_nodes())
        # 还可以添加更多信息

    # 在您的函数或脚本中 调用这些函数
    display_graph_info(GB_graph)


    end_time4 = time.perf_counter()
    #print("后续操作：", end_time4-start_time4, "秒")
    total_time += end_time4-start_time4
    print("总时间：", total_time, "秒")


    return new_f,new_C,total_time  # 返回处理后的数据



def gb_center_radius(new_C):
    length = len(new_C)
    gb_data = []
    for i in range(length):
        granular_ball = new_C[i]
        center, radius = calculate_center_and_radius(granular_ball)
        #print("center:", center)
        gb_data.append([center, radius])
    return gb_data




