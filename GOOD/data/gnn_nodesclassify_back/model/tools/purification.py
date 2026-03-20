
def find_major(nums):
    candidate = 0  # 候选众数
    count = 0  # 候选众数的出现次数

    # 遍历数组
    for num in nums:
        if num[-2] != -1:#不是测试集
            if count == 0:
                candidate = num[-2]
                count = 1
            elif num[-2] == candidate:
                count += 1
            else:
                count -= 1

    return candidate

def purification(C):
    new_clusters = []
    for cluster in C:  # 遍历每个球簇
        data = cluster[0]  # 数据信息
        #FL_matrix = cluster[1]  # Floyd矩阵信息 （已不用）
        D_dict = cluster[1]  # 度字典
        pur = cluster[2]  # 粒球纯度

        major_label = find_major(data)  # 粒球中最多的标签 作为粒球的标签
        #major_label_num = find_major_num(data, major_label)  # 最多的标签的点个数
        #if major_label_num > 0.8 * len(data):  # 数量大于80%则加入新簇
        # temp_C = [data, FL_matrix, D_dict, major_label]
        temp_C = [data, D_dict, pur, major_label]
        new_clusters.append(temp_C)

    return new_clusters
