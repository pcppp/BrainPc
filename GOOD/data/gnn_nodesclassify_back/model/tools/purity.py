import numpy as np
from .find_major import find_major
from .find_major_num import find_major_num
def purity(Ci, labels):
    length = len(Ci[0])
    if length == 1:
        purity = 1
        return purity

    data = Ci[0]  #点信息
    # Ci[0] 包含该球的所有节点索引
    # index(0,3)
    #node_indices = index
    # 提取这些节点的标签
    #node_labels = labels[node_indices]
    # 计算最多的标签占比
    major_label = find_major(data)  # 最多的标签
    major_label_num = find_major_num(data, major_label)  # 最多的标签的点个数
    purity = major_label_num / length if major_label_num else 1.0
    return purity

