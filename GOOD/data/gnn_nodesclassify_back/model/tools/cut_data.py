

def cut_data(data, matrix):
    # 获取前100个元素
    cut_data = data[:200]

    # 获取前100行和前100列的矩阵
    cut_matrix = matrix[:200, :200]

    return cut_data, cut_matrix