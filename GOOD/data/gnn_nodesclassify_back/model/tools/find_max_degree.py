import numpy as np
ini = float('inf')
from operator import itemgetter

# 用于寻找分裂原点
def find_max_degree(degree_dict):

    sorted_dict_desc = dict(sorted(degree_dict.items(), key=itemgetter(1), reverse=True))
    #print("sorted_dict_desc",sorted_dict_desc)
    # 返回两个最大度节点的索引
    max_index = [list(sorted_dict_desc.keys())[0], list(sorted_dict_desc.keys())[1]]
    max_value = 0
    return max_value, max_index


def find_max_degree_by2(matrix,degree_dict):
    # 初始化最大值为负无穷大
    max_value = float('-inf')
    # 记录最大值的索引
    max_index = None
    #print("degree_dict(find max)",degree_dict)
    # 遍历矩阵中的每个元素
    for i in degree_dict:
        for j in degree_dict:
            # 如果当前元素为预设的ini值,则直接返回ini和该元素的索引
            if matrix[i][j] == ini:
                return ini, (i, j)
    # 如果当前元素不是无穷

    sorted_dict_desc = dict(sorted(degree_dict.items(), key=itemgetter(1), reverse=True))
    #print("sorted_dict_desc",sorted_dict_desc)
    # 返回两个最大度节点的索引
    max_index = [list(sorted_dict_desc.keys())[0], list(sorted_dict_desc.keys())[1]]
    max_value = 0
    return max_value, max_index
