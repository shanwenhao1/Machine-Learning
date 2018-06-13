#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/8
# @Author  : Wenhao Shan
# @Dsc     : K-Means Clustering ++

import numpy as np
import random
from PythonMachineLearning import functionUtils as FTool
from PythonMachineLearning.Code.Chapter10.k_means import distance, k_means_function

FLOAT_MAX = 1e100   # 设置一个较大的值作为初始化的最小距离


def nearst(point: np.mat, cluster_centers: np.mat):
    """
    计算point与cluster_centers之间的最小距离
    :param point: 当前的样本点
    :param cluster_centers: 聚类中心
    :return:
    """
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]     # 当前已经初始化的聚类中心
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist


def get_centroids(points: np.mat, k: int):
    """
    K-Means++的初始化聚类中心方法
    :param points: 样本
    :param k: 聚类中心的个数
    :return:
    """
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k, n)))
    # 1、随机选择一个样本点作为第一个聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0, ] = np.copy(points[index, ])
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]

    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearst(points[j, ], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all 之间的随机值
        sum_all *= random.random()
        # 6、依概率获得距离最远的样本点作为聚类中心
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j, ])
            break
    return cluster_centers


def k_means_pp():
    """
    K-Means ++ 实现
    :return:
    """
    k = 4   # 聚类中心的个数
    file_path = "data.txt"
    # 1、导入数据
    print("---------- 1.load data ------------")
    data, _, _ = FTool.LoadData(file_name=file_path).load_data(feature_end=0, need_label_length=True, need_list=True)
    x_data = [_data[0] for _data in data]
    y_data = [_data[1] for _data in data]
    with FTool.PaintingWithList(name="K-Means") as paint:
        paint.painting_simple_list(x_data, y_data)
    # 2、KMeans++的聚类中心初始化方法
    print("---------- 2.K-Means++ generate centers ------------")
    data = np.mat(data)
    centroids = get_centroids(data, k)
    # 3、聚类计算
    print("---------- 3.k-Means ------------")
    sub_center = k_means_function(data, k, centroids)
    # 4、保存所属的类别文件
    print("---------- 4.save subCenter ------------")
    with FTool.SaveModel(file_name="sub_pp") as save_model:
        save_model.save_model_mul(sub_center)
    # 5、保存聚类中心
    print("---------- 5.save centroids ------------")
    with FTool.SaveModel(file_name="center_pp") as save_model:
        save_model.save_model_mul(centroids)


if __name__ == '__main__':
    k_means_pp()
