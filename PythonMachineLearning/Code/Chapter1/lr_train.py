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
from PythonMachineLearning.functionUtils import sig, PaintingPicture


def load_data(file_name: str):
    """
    导入数据
    :param file_name:
    :return:
    """
    f = open(file_name)
    feature_data = list()
    label_data = list()
    for line in f.readlines():
        feature_tmp = list()
        lable_tmp = list()
        lines = line.strip().split("\t")
        feature_tmp.append(1)   # 偏置项
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        lable_tmp.append(float(lines[-1]))

        feature_data.append(feature_tmp)
        label_data.append(lable_tmp)
    f.close()
    with PaintingPicture(name="LR Point") as paint:
        paint.painting(np.mat(feature_data), np.mat(label_data))
    return np.mat(feature_data), np.mat(label_data)


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


def save_model(file_name: str, w: np.mat):
    """
    保存最终的模型
    :param file_name:
    :param w:   LR模型的权重
    :return:
    """
    m = np.shape(w)[0]
    f_w = open(file_name, "w")
    w_array = [str(w[i, 0]) for i in range(m)]
    f_w.write("\t".join(w_array))
    f_w.close()


def TrainOfLR():
    """
    训练并保存模型
    :return:
    """
    print("------------------------1. Load Data-----------------------")
    feature, label = load_data("data.txt")

    print("------------------------2. Training-----------------------")
    w = lr_train_bgd(feature, label, 1000, 0.01)

    print("------------------------3. Save Model-----------------------")
    save_model("weights", w)


if __name__ == '__main__':
    TrainOfLR()
