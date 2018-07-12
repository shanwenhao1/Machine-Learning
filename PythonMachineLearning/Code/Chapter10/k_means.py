#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/8
# @Author  : Wenhao Shan
# @Dsc     : K-Means Clustering

import numpy as np
from utils.errors import ActionError
from PythonMachineLearning import functionUtils as FTool


def rand_cent(data: np.mat, k: int):
    """
    随机初始化聚类中心, 但是由于是事先随机选取中心(对结果影响较大), 多次运行结果差异较大
    :param data: 训练数据, m * n(m个样本, n个特征)
    :param k: 类别个数
    :return: centroids(mat): 聚类中心
    """
    n = np.shape(data)[1]   # 属性的个数
    centroids = np.mat(np.zeros((k, n)))    # 初始化k个聚类中心
    for j in range(n):  # 初始化聚类中心每一维的坐标
        min_j = np.min(data[:, j])
        range_j = np.max(data[:, j]) - min_j
        # 在最大值和最小值之间随机初始化, np.random.rand(k, 1)随机生成k维0-1的矩阵
        centroids[:, j] = min_j * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * range_j
    return centroids


def distance(vec_a, vec_b):
    """
    计算vector A 和vector B之间的欧氏距离平方
    :param vec_a:
    :param vec_b:
    :return:
    """
    dist = (vec_a - vec_b) * (vec_a - vec_b).T
    return dist[0, 0]


def k_means_function(data: np.mat, k: int, centroids: np.mat):
    """
    根据K-Means算法求解聚类中心
    :param data: 训练数据
    :param k: 类别个数
    :param centroids: 随机初始化的聚类中心
    :return: centroids(mat): 训练完成的聚类中心
              sub_center(mat): 每一个样本所属的类别
    """
    m, n = np.shape(data)   # 样本个数和特征的维度
    sub_center = np.mat(np.zeros((m, 2)))   # 初始化每一个样本所属的类别\
    change = True   # 判断是否需要重新计算聚类中心
    while change:
        change = False  # 重置
        for i in range(m):
            min_dist = np.inf   # 设置样本与聚类中心之间的最小的距离, 初始值为正无穷
            min_index = 0   # 所属的类别
            for j in range(k):
                # 计算i和每个聚类中心之间的距离
                dist = distance(data[i, ], centroids[j, ])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            # 判断是否需要改变
            if sub_center[i, 0] != min_index:
                change = True
                sub_center[i, ] = np.mat([min_index, min_dist])
        # 重新计算聚类中心
        for j in range(k):
            sum_all = np.mat(np.zeros((1, n)))
            r = 0   # 每个类别中的样本的个数
            for i in range(m):
                if sub_center[i, 0] == j:   # 计算第j个类别
                    sum_all += data[i, ]
                    r += 1
            for z in range(n):
                try:
                    centroids[j, z] = sum_all[0, z] / r
                except ActionError("r is zero"):
                    pass
    return sub_center


def k_means(k_class: int):
    """
    The K-Means Function
    :param k_class: 聚类中心的个数
    :return:
    """
    k = k_class  # 聚类中心的个数
    file_path = "data.txt"
    # 1、导入数据
    print("---------- 1.load data ------------")
    data, _, _ = FTool.LoadData(file_name=file_path).load_data(feature_end=0, need_label_length=True, need_list=True)
    x_data = [_data[0] for _data in data]
    y_data = [_data[1] for _data in data]
    with FTool.PaintingWithList(name="K-Means") as paint:
        paint.painting_simple_list(x_data, y_data)
    # 2、随机初始化k个聚类中心
    print("---------- 2.random center ------------")
    data = np.mat(data)
    centroids = rand_cent(data, k)
    # 3、聚类计算
    print("---------- 3.kmeans ------------")
    sub_center = k_means_function(data, k, centroids)
    # 4、保存所属的类别文件
    print("---------- 4.save subCenter ------------")
    with FTool.SaveModel(file_name="sub") as save_model:
        save_model.save_model_mul(sub_center)
    # 5、保存聚类中心
    print("---------- 5.save centroids ------------")
    with FTool.SaveModel(file_name="center") as save_model:
        save_model.save_model_mul(centroids)


if __name__ == '__main__':
    k_means(4)
