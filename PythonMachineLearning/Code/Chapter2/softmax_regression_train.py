#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/22
# @Author  : Wenhao Shan

import numpy as np
from PythonMachineLearning.functionUtils import PaintingWithMat, LoadData, SaveModel


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
        weights += (alpha / m) * feature_data.T * err  # 权重修正
        i += 1
    return weights


def sr_train():
    """
    训练并保存SR模型
    :return:
    """
    # 1. 导入训练数据
    print("--------------- 1. load data-----------------")
    feature, label, k = LoadData(file_name="SoftInput.txt", label_type="int").load_data(
        offset=1, need_label_length=True)
    label = label.T
    with PaintingWithMat(name="SoftMax Point") as paint:
        paint.painting_with_offset(feature, label)
    # 2. 训练Softmax模型
    print("--------------- 2. training-----------------")
    weights = gradientAscent(feature, label, k, 10000, 0.4)
    # 3. 保存最终的模型
    print("--------------- 3. save model-----------------")
    with SaveModel("weights") as save_model:
        save_model.save_model_mul(weights)


if __name__ == '__main__':
    sr_train()
