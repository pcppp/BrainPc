import numpy as np


def calculate_center_and_radius(granular_ball):
    data = np.array(granular_ball[0])
    center = data[:, 1:3].mean(axis=0)  # 平均中心
    # 将半径r定义为平均距离而不是最大或最小距离的主要原因是,颗粒球的大小不容易受到离群样本的影响。
    radius = np.mean((((data[:, 1:3] - center) ** 2).sum(axis=1) ** 0.5))  # 所有数据距离平均中心的平均距离
    return center, radius
