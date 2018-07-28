#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/16
# @Author  : Wenhao Shan
# @Dsc     : PersonRank Recommend

import numpy as np
from PythonMachineLearning import functionUtils as FTool


def generate_dict(data_tmp: np.mat):
    """
    将用户-商品矩阵转换成二部图的表示
    :param data_tmp: 用户商品矩阵
    :return: data_dict(dict): 图的表示
    """
    m, n = np.shape(data_tmp)

    data_dict = dict()
    # 对每一个用户生成节点
    for i in range(m):
        tmp_dict = {"D_" + str(j): data_tmp[i, j] for j in range(n) if data_tmp[i, j] != 0}
        data_dict["U_" + str(i)] = tmp_dict

    # 对每一个商品生成节点
    for j in range(n):
        tmp_dict = {"U_" + str(i): data_tmp[i, j] for i in range(m) if data_tmp[i, j] != 0}
        data_dict["D_" + str(j)] = tmp_dict
    return data_dict


def recommend(data_dict: dict, rank: dict, user: str):
    """
    得到最终的推荐列表
    :param data_dict: 用户-商品的二部图表示
    :param rank: 打分的结果
    :param user: 用户
    :return: result(dict): 推荐结果
    """
    item_dict = dict()
    # 1、用户user已打过分的项
    items = [k for k in data_dict[user].keys()]

    # 2、从rank取出商品的打分
    for k in rank.keys():
        if k.startswith("D_"):  # 商品
            if k not in items:  # 排除已经互动过的商品
                item_dict[k] = rank[k]
    # item_dict = {k: rank[k] for k in rank.keys() if k.startswith("D_") and k not in items}

    # 3、按打分的降序排序
    result = sorted(item_dict.items(), key=lambda d: d[1], reverse=True)
    return result


def personal_rank(data_dict: dict, alpha: float, user: str, max_cycles):
    """
    利用PersonalRank打分
    :param data_dict: 用户-商品的二部图表示
    :param alpha: 跳出当前链接的概率(0-1之间)
    :param user: 指定用户
    :param max_cycles: 最大的迭代次数
    :return: rank(dict): 打分的列表
    """
    # 1、初始化打分
    rank = {x: 0 for x in data_dict.keys()}
    # 初始化初始转移概率, 其中user(用户)的概率为1
    rank[user] = 1  # 从user开始游走(附上1的概率, 通常整个图的概率加起来为1)

    # 2、迭代
    step = 0
    while step < max_cycles:
        # 初始化二部图, 都为0
        tmp = {x: 0 for x in data_dict.keys()}

        for k, v in data_dict.items():
            for j in v.keys():
                if j not in tmp:
                    tmp[j] = 0
                # PR公式更新概率转移二部图
                tmp[j] += alpha * rank[k] / (1.0 * len(v))
                if j == user:
                    tmp[j] += (1 - alpha)

        # 判断是否收敛完毕
        check = [tmp[k] - rank[k] for k in tmp.keys()]  # PR概率转移变化列表
        if sum(check) <= 0.0001:
            break
        rank = tmp
        if step % 20 == 0:
            print("iter: ", step)
        step += 1
    return rank


def personal_rank_run():
    """
    Personal Rank run
    :return:
    """
    # 1、导入用户商品矩阵
    print("------------ 1.load data -------------")
    data_mat = FTool.LoadData(file_name="data.txt").load_data_with_none()
    # 2、将用户商品矩阵转换成邻接表的存储(二部图)
    print("------------ 2.generate dict --------------")
    data_dict = generate_dict(data_mat)
    # 3、利用PersonalRank计算
    print("------------ 3.PersonalRank --------------")
    rank = personal_rank(data_dict, 0.85, "U_0", 500)
    # 4、根据rank结果进行商品推荐
    print("------------ 4.recommend -------------")
    result = recommend(data_dict, rank, "U_0")
    print(result)


if __name__ == '__main__':
    personal_rank_run()
