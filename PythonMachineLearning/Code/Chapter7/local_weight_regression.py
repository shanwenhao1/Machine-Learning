#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/3
# @Author  : Wenhao Shan
# @Dsc     : Local Weight Regression training

import numpy as np
from PythonMachineLearning.functionUtils import PaintingWithList, get_list_from_mat, LoadData


def lwlr(feature: np.mat, label: np.mat, k: int):
    """
    局部加权线性回归
    :param feature: 特征
    :param label: 标签
    :param k: 核函数的系数
    :return: predict(mat): 最终的结果
    """
    m = np.shape(feature)[0]
    predict = np.zeros(m)
    weights = np.mat(np.eye(m))
    for i in range(m):
        for j in range(m):
            diff = feature[i, ] - feature[j, ]
            weights[j, j] = np.exp(diff * diff.T / (-2.0 * k ** 2))
        xTx = feature.T * (weights * feature)
        ws = xTx.I * (feature.T * (weights * label))
        predict[i] = feature[i, ] * ws
    return predict


def local_weight_regression_train():
    """
    局部加权线性回归的训练及预测
    :return:
    """
    # 1、导入数据集
    # feature, label = LRTrain.load_data("data.txt")
    feature, label, _ = LoadData(file_name="data.txt").load_data(offset=1, need_label_length=True, need_list=True)
    with PaintingWithList(name="Linear Regression Training") as paint:
        paint.painting_with_offset(feature, label)
    feature = np.mat(feature)
    label = np.mat(label).T

    predict = lwlr(feature, label, 0.002)
    m = np.shape(predict)[0]
    for i in range(m):
        print(feature[i, 1], predict[i])
    x_data = get_list_from_mat(feature, 1, True)
    y_data = list(predict)
    with PaintingWithList(name="Local Weight Regression") as paint:
        paint.painting_no_offset(x_data, y_data)


if __name__ == '__main__':
    local_weight_regression_train()
