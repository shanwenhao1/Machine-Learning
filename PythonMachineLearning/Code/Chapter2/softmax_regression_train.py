#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/22
# @Author  : Wenhao Shan

import numpy as np
from PythonMachineLearning.functionUtils import PaintingWithLabel


def load_data(inputfile: str):
    """
    导入训练数据
    :param inputfile:
    :return:           feature_data(mat)特征
                        label_data(标签)
                        k(int)类别的个数
    """
    f = open(inputfile)
    feature_data = list()
    label_data = list()
    for line in f.readlines():
        feature_tmp = list()
        feature_tmp.append(1)   # 偏置顶
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_data.append(int(lines[-1]))

        feature_data.append(feature_tmp)

    f.close()
    with PaintingWithLabel(name="SoftMax Point") as paint:
        paint.painting_with_offset(np.mat(feature_data), np.mat(label_data).T)
    # set消除重复元素
    return np.mat(feature_data), np.mat(label_data).T, len(set(label_data))


def cost(err: np.mat, label_data: np.mat):
    """
    计算损失函数值
    :param err: exp的值
    :param label_data:  标签的值
    :return:                (float) 损失函数的值
    """
    m = np.shape(err)[0]
    sum_cost = 0.0
    for i in range(m):
        if err[i, label_data[i, 0]] / np.sum(err[i, :]) > 0:
            sum_cost -= np.log(err[i, label_data[i, 0]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost / m


def gradientAscent(feature_data: np.mat, label_data: np.mat, k: int, maxCycle: int, alpha: float):
    """
    利用梯度下降法训练softmax模型
    :param feature_data:    特征
    :param label_data:  标签
    :param k:   类别的个数
    :param maxCycle:    最大的迭代次数
    :param alpha:   学习率(梯度下降法中的学习步长alpha)
    :return:                (mat)   权重
    """
    m, n = np.shape(feature_data)
    weights = np.mat(np.ones((n, k)))     # 权重的初始化
    i = 0
    while i <= maxCycle:
        # 这里相当于
        err = np.exp(feature_data * weights)
        if i % 100 == 0:
            # 计算损失函数值
            print("\t-----------iter: ", i, ", cost: ", cost(err, label_data))
        rowSum = -err.sum(axis=1)   # 将矩阵每一行(axis=1)元素相加
        rowSum = rowSum.repeat(k, axis=1)   # repeat elements of an array, 为了使err / rowSum的分母都相同
        err = err / rowSum
        for x in range(m):
            # 矩阵所属类型对应列数置为正数
            err[x, label_data[x, 0]] += 1
        weights = weights + (alpha / m) * feature_data.T * err  # 权重修正
        i += 1
    return weights


def save_model(file_name: str, weights: np.mat):
    """
    保存最终的模型
    :param file_name:
    :param weights:
    :return:
    """
    f_w = open(file_name, "w")
    m, n = np.shape(weights)
    for i in range(m):
        w_tmp = [str(weights[i, j]) for j in range(n)]
        f_w.write("\t".join(w_tmp) + "\n")
    f_w.close()


def TrainOfSR():
    """
    训练并保存SR模型
    :return:
    """
    inputfile = "SoftInput.txt"
    # 1. 导入训练数据
    print("--------------- 1. load data-----------------")
    feature, label, k = load_data(inputfile)
    # 2. 训练Softmax模型
    print("--------------- 2. training-----------------")
    weights = gradientAscent(feature, label, k, 10000, 0.4)
    # 3. 保存最终的模型
    print("--------------- 3. save model-----------------")
    save_model("weights", weights)


if __name__ == '__main__':
    TrainOfSR()
