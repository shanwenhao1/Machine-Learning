#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/21
# @Author  : Wenhao Shan

"""
具体步骤：
            1. 导入训练数据
            2. 利用梯度下降法对训练数据进行训练, 以得到Logistic Regression算法的模型, 即模型中的权重
            3. 将权重输出到文件weights中
"""
import numpy as np
from PythonMachineLearning.functionUtils import sig, PaintingWithMat, LoadData, SaveModel


def lr_train_bgd(feature: np.mat, label: np.mat, maxCycle: int, alpha: float):
    """
    利用梯度下降法训练LR模型
    :param feature: 特征
    :param label:   标签
    :param maxCycle:    最大迭代次数
    :param alpha:   学习率
    :return:            w(mat)  权重
    """
    n = np.shape(feature)[1]    # 特征个数
    w = np.mat(np.ones((n, 1)))  # 初始化权重
    i = 0
    while i <= maxCycle:
        i += 1
        h = sig(feature * w)    # 计算Sigmoid值
        err = label - h
        if i % 100 == 0:
            print("\t--------------------------iter=" + str(i) + ", train error rate= " + str(error_rate(h, label)))
        w += alpha * feature.T * err  # 权重修正
    return w


def error_rate(h: np.mat, label: np.mat):
    """
    计算当前的损失函数值
    :param h:   预测值
    :param label:   实际值
    :return:    (float) 错误率
    """
    m = np.shape(h)[0]

    sum_err = 0.0
    for i in range(m):
        z = h[i, 0]
        if h[i, 0] > 0 and (1 - h[i, 0]) > 0:
            sum_err -= (label[i, 0] * np.log(h[i, 0]) + (1 - label[i, 0]) * np.log(1 - h[i, 0]))
        else:
            sum_err -= 0
    return sum_err / m


def TrainOfLR():
    """
    训练并保存模型
    :return:
    """
    print("------------------------1. Load Data-----------------------")
    feature, label = LoadData(file_name="data.txt", feature_type="float", label_type="float").load_data(offset=1)
    with PaintingWithMat(name="LR Point") as paint:
        paint.painting_with_offset(feature, label)

    print("------------------------2. Training-----------------------")
    w = lr_train_bgd(feature, label, 1000, 0.01)

    print("------------------------3. Save Model-----------------------")
    with SaveModel("weights") as save_model:
        save_model.save_model(w)


if __name__ == '__main__':
    TrainOfLR()
