#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/14
# @Author  : Wenhao Shan
# @Dsc     : Recommendation based on user-based

import numpy as np


def cos_sim(x: np.mat, y: np.mat):
    """
    余弦相似度方法
    :param x: 以行向量的形式存储, 可以是用户或者商品
    :param y: 以行向量的形式存储, 可以是用户或者商品
    :return: x 和 y 之间的余弦相似度
    """
    numerator = x * y.T     # x和y之间的内积
    denominator = np.sqrt(x * x.T) * np.sqrt(y * y.T)
    return (numerator / denominator)[0, 0]


def similarity(data: np.mat):
    """
    计算矩阵中任意两行之间的余弦相似度
    :param data: 任意矩阵
    :return: w(mat): 任意两行之间的相似度
    """
    m = np.shape(data)[0]   # 用户的数量
    # 初始化相似度矩阵
    w = np.mat(np.zeros((m, m)))

    for i in range(m):
        for j in range(i, m):
            if j != i:
                # 计算任意两行之间的相似度
                w[i, j] = cos_sim(data[i, ], data[j, ])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w


def user_based_recommend(data: np.mat, w: np.mat, user: int):
    """
    基于用户相似性, 为用户推荐商品
    :param data: 用户商品矩阵
    :param w: 用户之间的相似度
    :param user: 用户的编号
    :return: predict(list): 推荐列表
    """
    m, n = np.shape(data)
    interaction = data[user, ]  # 用户user与商品信息

    # 1、找到用户user没有互动过的商品
    not_inter = list()
    for i in range(n):
        if interaction[0, i] == 0:  # 没有互动(评价)的商品
            not_inter.append(i)

    # 2、对没有互动过的商品进行预测(利用所有评价过该商品的用户的分数 * 用户之间的相似度进行评价)
    predict = dict()
    for x in not_inter:
        item = np.copy(data[:, x])  # 找到所有用户对商品x的互动信息
        for i in range(m):  # 对每一个用户
            if item[i, 0] != 0:     # 若该用户对商品x有过互动
                if x not in predict:
                    predict[x] = w[user, i] * item[i, 0]
                else:
                    predict[x] = predict[x] + w[user, i] * item[i, 0]
    # 3、按照预测的大小从大到小排序
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)


def top_k(predict: list, k: int):
    """
    为用户推荐前k个商品
    :param predict: 排好序的商品列表
    :param k: 推荐的商品个数
    :return: top_recommend(list): top_k个商品
    """
    len_result = len(predict)
    if k >= len_result:
        top_recommend = predict
    else:
        top_recommend = predict[: k]
    return top_recommend
