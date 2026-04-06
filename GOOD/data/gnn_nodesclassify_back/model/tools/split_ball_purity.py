from .find_max_degree import find_max_degree,find_max_degree_by2
from .purity import purity
from .split_2_co import split_2_co

import queue
ini = float('inf')
current_balls_num = 0

def split_ball_purity(graph, id_dict, C_queue, C, labels, total_degree_dict, total_balls_num,purity_threshold=1.0):
    new_clusters = []  # 存储分裂后的新球簇列表

    #print("分裂")
    global current_balls_num
    #print("current:前", current_balls_num)

    for cluster in C:  # 遍历每个球簇
        #如果当前求是粗分球且遍历过，则移出队列
        #if cluster[0][0][0] == C_queue.queue[0][0][0][0] and cluster[0][0][-1] == C_queue.queue[0][0][0][-1]:
        if cluster[0][0][-1] == C_queue.queue[0][0][0][-1] and cluster[0][-1][-1] == C_queue.queue[0][0][-1][-1]:
            #print("存在相同")
            C_queue.get()
        # if len(cluster[0])<3:
        #     # print("粗分得到的小于等于2的球",len(cluster))
        #     new_clusters.append(cluster)
        #     current_balls_num+=1
        #     continue
        if len(cluster[0]) == 1:
            # print("粗分得到的小于等于2的球",len(cluster))
            new_clusters.append(cluster)
            current_balls_num += 1
            if current_balls_num+C_queue.qsize() >= total_balls_num:
                return new_clusters
            continue
        value, index = find_max_degree(cluster[1])# 度字典

        #print("连同")
        cluster1, cluster2 = split_2_co(graph, id_dict, cluster, index, total_degree_dict)  # 分裂得到的两个子球簇
        #print("co完")

        #判断是否继续分裂
        if current_balls_num + C_queue.qsize() >= total_balls_num:
            new_clusters.append(cluster)
            return new_clusters
        # 检查是否继续分裂
        if len(cluster1[0]) < 3 and len(cluster2[0]) < 3:
            new_clusters.append(cluster)
            current_balls_num += 1
            continue
        if len(cluster1[0]) < 3:
            if current_balls_num+C_queue.qsize() >= total_balls_num:
                new_clusters.append(cluster)
                current_balls_num += 1
                return new_clusters
            new_clusters.append(cluster1)  # 将子球簇1加入新球簇列表
            new_clusters.extend(split_ball_purity(graph, id_dict, C_queue, [cluster2], labels, total_degree_dict, total_balls_num, purity_threshold))  # 对子球簇2进行递归分裂
            current_balls_num += 1
            continue
        if len(cluster2[0]) < 3:
            if current_balls_num+C_queue.qsize() >= total_balls_num:
                new_clusters.append(cluster)
                current_balls_num += 1
                return new_clusters
            new_clusters.append(cluster2)  # 将子球簇2加入新球簇列表
            new_clusters.extend(split_ball_purity(graph, id_dict, C_queue, [cluster1], labels, total_degree_dict, total_balls_num, purity_threshold))  # 对子球簇1进行递归分裂
            current_balls_num += 1
            continue

        purity1 = purity(cluster1, labels)
        purity2 = purity(cluster2, labels)

        if purity1 < purity_threshold and purity2 < purity_threshold:
            #分裂出的两个粒球先分裂纯度小的
            if cluster1[-1] <= cluster2[-1]:
                new_clusters.extend(split_ball_purity(graph, id_dict, C_queue, [cluster1], labels, total_degree_dict, total_balls_num, purity_threshold))
                new_clusters.extend(split_ball_purity(graph, id_dict, C_queue, [cluster2], labels, total_degree_dict, total_balls_num, purity_threshold))
            else:
                new_clusters.extend(split_ball_purity(graph, id_dict, C_queue, [cluster2], labels, total_degree_dict, total_balls_num, purity_threshold))
                new_clusters.extend(split_ball_purity(graph, id_dict, C_queue, [cluster1], labels, total_degree_dict, total_balls_num, purity_threshold))
        elif purity1 == purity_threshold and purity2 == purity_threshold:
            if current_balls_num+C_queue.qsize() >= total_balls_num:
                new_clusters.append(cluster)
                current_balls_num += 1
                return new_clusters
            new_clusters.append(cluster1)
            new_clusters.append(cluster2)
            current_balls_num += 2

        elif purity1 < purity_threshold and purity2 == purity_threshold:
            if current_balls_num+C_queue.qsize() >= total_balls_num:
                new_clusters.append(cluster)
                current_balls_num += 1
                return new_clusters
            new_clusters.extend(split_ball_purity(graph, id_dict, C_queue, [cluster1], labels, total_degree_dict, total_balls_num, purity_threshold))
            new_clusters.append(cluster2)
            current_balls_num += 1
        else:
            if current_balls_num+C_queue.qsize() >= total_balls_num:
                new_clusters.append(cluster)
                current_balls_num += 1
                return new_clusters
            new_clusters.extend(split_ball_purity(graph, id_dict, C_queue, [cluster2], labels, total_degree_dict, total_balls_num, purity_threshold))
            new_clusters.append(cluster1)
            current_balls_num += 1
        #print("局部完")

    #print("current:后", current_balls_num)
    return new_clusters


#非递归二分裂
def split_ball_purity_unrRcursion(graph, id_dict, C, total_degree_dict, total_balls_num, purity_threshold=1):
    cur_ball_num = len(C)
    while True:
        # 对粒球按照类别数排序
        # C.sort(key=lambda x: len(x[0])*(1-x[-1]), reverse=True) #考虑纯度和大小
        C.sort(key=lambda x: x[-1]) #只考虑纯度
        if cur_ball_num >= total_balls_num or C[0][-1] >= purity_threshold:
            break

        GB = C.pop(0)
        value, index = find_max_degree(GB[1])  # 度字典
        # 选择粒球序列的第一个来分裂（第一个纯度最差）
        cluster1, cluster2 = split_2_co(graph, id_dict, GB, index, total_degree_dict)
        # 将分裂得到的两个子粒球加入粒球序列并排序
        temp_num = -1
        if len(cluster1) != 0:
            C.append(cluster1)
            temp_num += 1
        if len(cluster2) != 0:
            C.append(cluster2)
            temp_num += 1

        cur_ball_num += temp_num

    return C


#再次二分裂
def split_ball_further(graph, id_dict, C, total_degree_dict, total_balls_num,purity_threshold=1.0):
    cur_ball_num = len(C)

    while True:
        if cur_ball_num >= total_balls_num:
            break
        C.sort(key=lambda x: len(x[0]), reverse=True)
        GB = C.pop(0)
        value, index = find_max_degree(GB[1])  # 度字典
        # 选择粒球序列的第一个来分裂（第一个纯度最差）
        cluster1, cluster2 = split_2_co(graph, id_dict, GB, index, total_degree_dict)
        # 将分裂得到的两个子粒球加入粒球序列并排序
        temp_num = -1
        if len(cluster1) != 0:
            C.append(cluster1)
            temp_num += 1
        if len(cluster2) != 0:
            C.append(cluster2)
            temp_num += 1

        cur_ball_num += temp_num

    return C


def split_ball_purity_by2(C, labels, total_degree_dict, total_Fl, purity_threshold=1.0):
    new_clusters = []  # 存储分裂后的新球簇列表
    # print("分裂")

    for cluster in C:  # 遍历每个球簇
        # print("cluster[0]",len(cluster[0]))
        # print("进循环")

        value, index = find_max_degree_by2(total_Fl,total_degree_dict)  # floyd, 度字典
        # print("找到最大度")
        if value == ini:
            # print("不连同")
            cluster1, cluster2 = split_2_unco(cluster, index, total_degree_dict, total_Fl)  # 分裂得到的两个子球簇
            # print("unco完")
            if len(cluster1[0]) < 3 and len(cluster2[0]) < 3:
                new_clusters.append(cluster)
                continue
            if len(cluster1[0]) < 3:
                new_clusters.append(cluster1)  # 将子球簇1加入新球簇列表
                new_clusters.extend(split_ball_purity([cluster2], labels, total_degree_dict, total_Fl))  # 对子球簇2进行递归分裂
                continue
            if len(cluster2[0]) < 3:
                new_clusters.append(cluster2)  # 将子球簇2加入新球簇列表
                new_clusters.extend(split_ball_purity([cluster1], labels, total_degree_dict, total_Fl))  # 对子球簇1进行递归分裂
                continue
            new_clusters.extend(split_ball_purity([cluster1, cluster2], labels, total_degree_dict, total_Fl,
                                                  purity_threshold))  # 对子球簇1和子球簇2进行递归分裂
        else:
            # print("连同")
            cluster1, cluster2 = split_2_co(cluster, index, total_degree_dict, total_Fl)  # 分裂得到的两个子球簇
            # print("co完")

            # 检查是否继续分裂
            if len(cluster1[0]) < 3 and len(cluster2[0]) < 3:
                new_clusters.append(cluster)
                continue
            if len(cluster1[0]) < 3:
                new_clusters.append(cluster1)  # 将子球簇1加入新球簇列表
                new_clusters.extend(split_ball_purity([cluster2], labels, total_degree_dict, total_Fl))  # 对子球簇2进行递归分裂
                continue
            if len(cluster2[0]) < 3:
                new_clusters.append(cluster2)  # 将子球簇2加入新球簇列表
                new_clusters.extend(split_ball_purity([cluster1], labels, total_degree_dict, total_Fl))  # 对子球簇1进行递归分裂
                continue

            purity1 = purity(cluster1, labels)
            purity2 = purity(cluster2, labels)

            if purity1 < purity_threshold and purity2 < purity_threshold:
                new_clusters.extend(
                    split_ball_purity([cluster1], labels, total_degree_dict, total_Fl, purity_threshold))
                new_clusters.extend(
                    split_ball_purity([cluster2], labels, total_degree_dict, total_Fl, purity_threshold))
            elif purity1 == purity_threshold and purity2 == purity_threshold:
                new_clusters.append(cluster1)
                new_clusters.append(cluster2)
            elif purity1 < purity_threshold and purity2 == purity_threshold:
                new_clusters.extend(
                    split_ball_purity([cluster1], labels, total_degree_dict, total_Fl, purity_threshold))
                new_clusters.append(cluster2)
            else:
                new_clusters.extend(
                    split_ball_purity([cluster2], labels, total_degree_dict, total_Fl, purity_threshold))
                new_clusters.append(cluster1)
        # print("局部完")
    return new_clusters