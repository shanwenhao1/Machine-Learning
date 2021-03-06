#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/3
# @Author  : Wenhao Shan
# @Dsc     : Linear Regression training

import numpy as np
from PythonMachineLearning.functionUtils import PaintingWithList, LoadData, SaveModel


def get_error(feature: np.mat, label: np.mat, w: np.mat):
    """
    计算误差
    :param feature: 特征
    :param label: 标签
    :param w: 线性回归模型的参数
    :return: 损失函数值
    """
    return (label - feature * w).T * (label - feature * w) / 2


def first_derivative(feature: np.mat, label: np.mat, w: np.mat):
    """
    计算一阶导函数的值, 平方差损失函数
    :param feature: 特征
    :param label: 标签
    :param w:
    :return: g(mat): 一阶导函数值
    """
    m, n = np.shape(feature)
    g = np.mat(np.zeros((n, 1)))
    for i in range(m):
        err = label[i, 0] - feature[i, ] * w
        for j in range(n):
            g[j, ] -= err * feature[i, j]
    return g


def second_derivative(feature: np.mat):
    """
    计算二阶导函数的值
    :param feature: 特征
    :return: G(mat): 二阶导函数值
    """
    m, n = np.shape(feature)
    G = np.mat(np.zeros((n, n)))
    for i in range(m):
        x_left = feature[i, ].T
        x_right = feature[i, ]
        # 矩阵相乘
        G += np.dot(x_left, x_right)
    return G


def get_min_m(feature: np.mat, label: np.mat, sigma: float, delta: float, d: np.mat, w: np.mat, g: np.mat):
    """
    计算步长中最小的值m
    :param feature: 特征
    :param label: 标签
    :param sigma:
    :param delta:
    :param d: 负的一阶导数除以二阶导数值
    :param w:
    :param g: 损失函数的一阶导数值
    :return: m(int): 最小m值
    """
    m = 0
    while True:
        w_new = w + pow(sigma, m) * d
        # 这里函数f(x)为平方损失函数, 即get_error()
        left = get_error(feature, label, w_new)
        right = get_error(feature, label, w) + delta * pow(sigma, m) * g.T * d
        if left <= right:
            break
        else:
            m += 1
    return m


def newton(feature: np.mat, label: np.mat, iter_max: int, sigma: float, delta: float):
    """
    全局牛顿法(具有二阶收敛性)
    :param feature: 特征
    :param label: 标签
    :param iter_max: 最大迭代次数
    :param sigma:
    :param delta:
    :return: w(mat): 回归系数
    """
    n = np.shape(feature)[1]
    w = np.mat(np.zeros((n, 1)))
    it = 0
    while it <= iter_max:
        g = first_derivative(feature, label, w)     # 一阶导数
        G = second_derivative(feature)  # 二阶导数
        d = - np.dot(G.I, g)    # 等价于d = -G.I * g, .I为逆矩阵
        m = get_min_m(feature, label, sigma, delta, d, w, g)    # 得到最小的非负整数m
        w += pow(sigma, m) * d  # 更新w
        if it % 10 == 0:
            print("\t------- iteration: ", it, " , error: ", get_error(feature, label, w)[0, 0])
        it += 1
    return w


def linear_regression_train():
    """
    LR training
    :return:
    """
    # 1、导入数据集
    print("----------- 1.load data ----------")
    feature, label, _ = LoadData(file_name="data.txt").load_data(offset=1, need_label_length=True, need_list=True)
    with PaintingWithList(name="Linear Regression Training") as paint:
        paint.painting_with_offset(feature, label)
    feature = np.mat(feature)
    label = np.mat(label).T
    # 2.1、最小二乘求解
    print("----------- 2.training ----------")
    # print "\t ---------- least_square ----------"
    # w_ls = least_square(feature, label)   # 最小二乘法对参数进行训练
    # 2.2、牛顿法
    print("\t ---------- newton ----------")
    w_newton = newton(feature, label, 50, 0.1, 0.5)
    # 3、保存最终的结果
    print("----------- 3.save result ----------")
    with SaveModel(file_name="weights") as save_model:
        save_model.save_model_mul(w_newton)


if __name__ == '__main__':
    linear_regression_train()
