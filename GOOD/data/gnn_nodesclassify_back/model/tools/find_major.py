import numpy as np


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