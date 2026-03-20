import random

def split_indices(indices, ratio1, ways=''): #ratio为百分制占比

    if ways =='random':
        # 随机打乱索引
        random.shuffle(indices)

        #计算列表长度
        total_len = len(indices)
        len_1 = total_len * (100-ratio1) // 100

        #划分索引列表
        list_1 = indices[:len_1]
        list_2 = indices[len_1:]
    else:
        total_len = len(indices)
        len_1 = total_len * (100-ratio1)//100
        #len_1 = total_len*ratio1//100
        # 划分索引列表
        list_1 = indices[:len_1]
        list_2 = indices[len_1:]


    return list_1, list_2
