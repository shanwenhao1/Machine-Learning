#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/21
# @Author  : Wenhao Shan
# @Dsc     :  DBSCAN Clustering

import math
import numpy as np
from PythonMachineLearning import functionUtils as FTool

MIN_PTS = 5  # 定义半径内的最少的数据点的个数


def distance(data: np.mat):
    """
    计算样本点之间的距离
    :param data: 样本
    :return: dis(mat): 样本点之间的距离
    """
    m, n = np.shape(data)
    dis = np.mat(np.zeros((m, m)))
    # 每一个样本点
    for i in range(m):
        # 该样本点与其他所有样本点(包括自己)之间的欧式距离
        for j in range(i, m):
            # 计算i和j之间的欧式距离
            tmp = 0
            for k in range(n):
                tmp += (data[i, k] - data[j, k]) * (data[i, k] - data[j, k])
            dis[i, j] = np.sqrt(tmp)
            # 样本j到i的距离记录, 减少计算量
            dis[j, i] = dis[i, j]
    return dis


def find_eps(distance_d: np.mat, eps: float):
    """
    找到距离小于或等于eps的样本的下标
    :param distance_d: 样本i与其他样本之间的距离
    :param eps: 半径的大小
    :return:
    """
    n = np.shape(distance_d)[1]
    ind = [j for j in range(n) if distance_d[0, j] <= eps]
    return ind


def dbscan_function(data: np.mat, eps: float, min_pts: int):
    """
    DBSCAN算法
    :param data: 需要聚类的数据集
    :param eps: 半径ε
    :param min_pts: 半径内的最少的数据点的个数
    :return: types(mat): 每个样本的类型: 核心点、边界点和噪音点
              sub_class(mat): 每个样本所属的类别
    """
    m = np.shape(data)[0]
    # 区分核心点1, 边界点0和噪音点-1
    types = np.mat(np.zeros((1, m)))
    sub_class = np.mat(np.zeros((1, m)))
    # 用来判断该点是否处理过, 0表示未处理过
    dealed = np.mat(np.zeros((m, 1)))
    # 计算每个数据点之间的距离(m * m的矩阵)
    dis = distance(data)
    # 用于标记类别
    number = 1
    # 对每一个点进行处理
    for i in range(m):
        # 只处理未处理的点
        if dealed[i, 0] != 0:
            continue
        # 找到第i个点到其他所有点的距离
        dis_i = dis[i, ]
        # 找到半径eps内的所有点对应的下标
        ind = find_eps(dis_i, eps)
        # 区分点的类型
        # 边界点
        if 1 < len(ind) < min_pts + 1:
            types[0, i] = 0
            sub_class[0, i] = 0
        # 噪音点
        if len(ind) == 1:
            types[0, i] = -1
            sub_class[0, i] = -1
            dealed[i, 0] = 1
        # 核心点
        if len(ind) >= min_pts + 1:
            types[0, i] = 1
            for x in ind:
                sub_class[0, x] = number
            # 判断核心点是否密度可达, 找到所有密度可达的点划分为同一类
            while len(ind) > 0:
                dealed[ind[0], 0] = 1
                dis_i = dis[ind[0], ]
                tmp = ind[0]
                del ind[0]  # 处理完删除该样本, 继续处理下一个
                ind_1 = find_eps(dis_i, eps)    # 找出该样本的ε邻域内的所有点, 并进行划分处理

                if len(ind_1) <= 1:     # 处理非噪音点, 相当于len(ind_1) > 1后面的才会执行
                    continue
                for x1 in ind_1:
                    sub_class[0, x1] = number
                if len(ind_1) >= min_pts + 1:
                    types[0, tmp] = 1
                else:
                    types[0, tmp] = 0

                for j in range(len(ind_1)):
                    if dealed[ind_1[j], 0] != 0:
                        continue
                    # 如果点未处理过则处理并划分类别
                    dealed[ind_1[j], 0] = 1
                    ind.append(ind_1[j])
                    sub_class[0, ind_1[j]] = number
            number += 1

    # 最后处理所有未分类的点为噪音点
    # 或者直接 ind_2 = ((sub_class == 0).nonzero())[1]
    none_class = sub_class == 0
    ind_2 = np.nonzero(none_class)[1]
    for x in ind_2:
        sub_class[0, x] = -1
        types[0, x] = -1
    return types, sub_class


def epsilon(data: np.mat, min_pts):
    """
    计算最佳半径
    :param data: 训练数据
    :param min_pts: 半径内的数据点的个数
    :return: eps(float): 半径
    """
    m, n = np.shape(data)
    # 取得矩阵中最大元素所在的行
    x_max = np.max(data, 0)
    # 取得矩阵中最小元素所在的行
    x_min = np.min(data, 0)
    # np.prod()将array内元素的乘积返回
    eps = ((np.prod(x_max - x_min) * min_pts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
    return eps


def dbscan():
    # 1、导入数据
    print("----------- 1、load data ----------")
    data, _ = FTool.LoadData(file_name="data.txt").load_data(feature_end=0)
    x_data = [_data[0, 0] for _data in data]
    y_data = [_data[0, 1] for _data in data]
    with FTool.PaintingWithList(name="DBSCAN Origin data") as paint:
        paint.painting_simple_list(x_data, y_data)
    # 2、计算最佳半径(得到ε值为1.384), 主要通过MIN_PTS影响ε值
    print("----------- 2、calculate eps ----------")
    eps = epsilon(data, MIN_PTS)
    # 3、利用DBSCAN算法进行训练
    print("----------- 3、DBSCAN -----------")
    types, sub_class = dbscan_function(data, eps, MIN_PTS)

    result = [sub_class[0, _position] for _position in range(np.shape(sub_class)[1])]
    with FTool.PaintingWithList(name="DBSCAN Result") as paint:
        paint.painting_list_with_label(x_data, y_data, result)
    # 4、保存最终的结果
    print("----------- 4、save result -----------")
    with FTool.SaveModel(file_name="types") as save_model:
        save_model.save_result_row(types)
    with FTool.SaveModel(file_name="sub_class") as save_model:
        save_model.save_result_row(sub_class)


if __name__ == '__main__':
    dbscan()
