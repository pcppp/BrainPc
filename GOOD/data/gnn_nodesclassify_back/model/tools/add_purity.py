
def find_major_num(nums, target):
    sum = 0
    num_len = 0
    for num in nums:
        if num[-2] == target:
            sum += 1
        if num[-2] != -1:
            num_len += 1

    return sum, num_len

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
def add_purity(C):
    for ball in C: #遍历每一个初始粒球
        nodes = ball[0] #每个粒球中的节点
        major_label = find_major(nodes)  # 最多的标签
        major_label_num, num_len = find_major_num(nodes, major_label)  # 最多的标签的点个数

        ball.append(float(major_label_num/num_len))


    return C