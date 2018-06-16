#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/14
# @Author  : Wenhao Shan
# @Dsc     : Recommendation based on item-based

import numpy as np
from PythonMachineLearning.Code.Chapter14.user_based_recommend import similarity, user_based_recommend, top_k
from PythonMachineLearning import functionUtils as FTool


def item_based_recommend(data: np.mat, w: np.mat, user: int):
    """
    基于商品相似度为用户user推荐商品
    :param data: 商品用户矩阵
    :param w: 商品与商品之间的相似性
    :param user: 用户的编号
    :return: predict(list): 推荐列表
    """
    m, n = np.shape(data)   # m: 商品数量, n: 用户数量
    interaction = data[:, user].T   # 用户user的互动商品信息

    # 1、找到用户user没有互动的商品
    not_inter = list()
    for i in range(n):
        if interaction[0, i] == 0:  # 用户未打分项
            not_inter.append(i)

    # 2、对没有互动过的商品进行预测
    predict = dict()
    for x in not_inter:
        item = np.copy(interaction)     # 获取用户user对商品的互动信息
        for j in range(m):  # 对每一个商品
            if item[0, j] != 0:     # 利用互动过的商品预测
                if x not in predict:
                    predict[x] = w[x, j] * item[0, j]
                else:
                    predict[x] = predict[x] + w[x, j] * item[0, j]
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)


def recommend_run():
    """
    系统过滤算法运行(包括基于用户和基于项的协同推荐)
    :return:
    """
    # 1、导入用户商品数据
    print("=======================User-based Recommend========================")
    print("------------ 1. load data ------------")
    data = FTool.LoadData(file_name="data.txt").load_data_with_none()
    # 2、计算用户之间的相似性
    print("------------ 2. calculate similarity between users -------------")
    w = similarity(data)
    # 3、利用用户之间的相似性进行推荐
    print("------------ 3. predict ------------")
    predict = user_based_recommend(data, w, 2)
    # 4、进行Top-K推荐
    print("------------ 4. top_k recommendation ------------")
    top_recommend = top_k(predict, 2)
    print(top_recommend)

    print("\n\n=======================User-based Recommend========================")
    data = data.T   # 将用户商品矩阵转换为商品用户矩阵
    w = similarity(data)
    predict = item_based_recommend(data, w, 2)
    # 4、进行Top-K推荐
    print("------------ 4. top_k recommendation ------------")
    top_recommend = top_k(predict, 2)
    print(top_recommend)


if __name__ == '__main__':
    recommend_run()
