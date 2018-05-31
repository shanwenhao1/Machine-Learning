#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/21
# @Author  : Wenhao Shan

import numpy as np
from PythonMachineLearning.Code.Chapter1 import lr_train
from PythonMachineLearning import functionUtils as FTool


def load_weight(w: str):
    """
    导入LR模型
    :param w:   权重所在的文件夹位置
    :return:
    """
    f = open(w)
    w = list()
    for line in f.readlines():
        lines = line.strip().split("\t")
        w_tmp = [float(x) for x in lines]
        w.append(w_tmp)
    f.close()
    return np.mat(w)


def load_data(file_name: str, n: int):
    """
    导入测试数据
    :param file_name:
    :param n:
    :return:
    """
    f = open(file_name)
    feature_data = list()
    for line in f.readlines():
        feature_tmp = list()
        lines = line.strip().split("\t")
        if len(lines) != n - 1:
            continue
        feature_tmp.append(1)
        for x in lines:
            feature_tmp.append(float(x))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data)


def predict(data: np.mat, w: np.mat):
    """
    对测试数据进行预测
    :param data:    测试数据的特征
    :param w:   模型的参数
    :return:    (mat)   最终的预测结果
    """
    h = FTool.sig(data * w.T)
    m = np.shape(h)[0]
    for i in range(m):
        if h[i, 0] < 0.5:
            h[i, 0] = 0.0
        else:
            h[i, 0] = 1.0
    return h


def save_result(file_name: str, result: np.mat):
    """
    保存最终的预测结果
    :param file_name:
    :param result:
    :return:
    """
    m = np.shape(result)[0]
    tmp = [str(result[i, 0]) for i in range(m)]
    f_result = open(file_name, "w")
    f_result.write("\t\n".join(tmp))
    f_result.close()


def TestOfLR():
    """
    测试逻辑回归算法
    :return:
    """
    lr_train.TrainOfLR()
    print("------------------------1. Load Model-----------------------")
    w = load_weight("weights")
    n = np.shape(w)[1]

    print("------------------------2. Load Data-----------------------")
    testData = load_data("test_data", n)

    print("------------------------3. Get Prediction-----------------------")
    h = predict(testData, w)

    print("------------------------4. Save Prediction-----------------------")
    save_result("result", h)


if __name__ == '__main__':
    TestOfLR()
