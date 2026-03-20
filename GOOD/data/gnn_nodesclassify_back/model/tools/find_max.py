import numpy as np
ini = float('inf')

# 用于寻找分裂原点
def find_max(matrix):
    # 初始化最大值为负无穷大
    max_value = float('-inf')
    # 记录最大值的索引
    max_index = None

    # 遍历矩阵中的每个元素
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            # 如果当前元素为预设的ini值,则直接返回ini和该元素的索引
            if matrix[i][j] == ini:
                return ini, (i, j)
            # 如果当前元素大于最大值,则更新最大值和索引
            if matrix[i][j] > max_value:
                max_value = matrix[i][j]
                max_index = np.array([i, j])

    return max_value, max_index
