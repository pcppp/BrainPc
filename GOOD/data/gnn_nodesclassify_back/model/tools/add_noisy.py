import numpy as np
import random
def add_noisy(data):

    class_nums = len(set(np.array(data.y)))  # 数据集种类数
    change_nums = 0 #噪声标签
    for i in range(0, len(data.train_mask)):
        if data.train_mask[i]:
            p = np.random.randint(0, 100)#0.05加躁率
            if p < 5:
                change_nums += 1
                data.y[i] = generate_random_except(data.y[i], class_nums)

    print("噪声：", change_nums)


    return data


# 生成除开标签以外的随机整数
def generate_random_except(excluded_number, class_nums):

    choice_list = []
    for i in range(0, class_nums):
        if i != excluded_number:
            choice_list.append(i)
    print(choice_list)
    return np.random.choice(choice_list)
