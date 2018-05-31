#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/23
# @Author  : Wenhao Shan
import numpy as np
import random
from PythonMachineLearning.Code.Chapter2.softmax_regression_train import TrainOfSR


def load_weights(weight_path: str):
    """
    导入训练好的Softmax模型
    :param weight_path:
    :return:           weights(mat)    将权重存到矩阵中
                        m(int)          权重的行数
                        n(int)          权重的列数
    """
    f = open(weight_path)
    w = list()
    for line in f.readlines():
        w_tmp = list()
        lines = line.strip().split("\t")
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)
    f.close()
    weights = np.mat(w)
    m, n = np.shape(weights)
    return weights, m, n


def load_data(num: int, m: int):
    """
    导入测试数据
    :param num: 生成的测试样本的个数
    :param m:   样本的维数
    :return:        (mat)   生成的测试样本
    """
    testDataSet = np.mat(np.ones((num, m)))
    for i in range(num):
        # 随机生成[-3, 3]之间的随机数
        testDataSet[i, 1] = random.random() * 6 - 3
        # 随机生成[0, 15]之间的随机数
        testDataSet[i, 2] = random.random() * 15
    return testDataSet


def predict(test_data: np.mat, weights: np.mat):
    """
    利用训练好的Softmax模型对测试数据进行预测
    :param test_data:   测试数据的特征
    :param weights: 模型的权重
    :return:    h.argmax(axis=1)所属的类别
    """
    h = test_data * weights
    return h.argmax(axis=1)     # 获得所属的类别


def save_result(file_name: str, result: np.matrix):
    """
    保存最终的预测结果
    :param file_name:
    :param result:
    :return:
    """
    f_result = open(file_name, "w")
    m = np.shape(result)[0]
    for i in range(m):
        f_result.write(str(result[i, 0]) + "\n")
    f_result.close()


def TestSR():
    """
    测试Softmax Regression
    :return:
    """
    TrainOfSR()
    print("---------- 1.Load Model ------------")
    w, m, n = load_weights("weights")
    print("---------- 2.Load Data ------------")
    test_data = load_data(4000, m)
    print("---------- 3.Prediction ------------")
    result = predict(test_data, w)
    print("---------- 4.Save Prediction ------------")
    save_result("result", result)


if __name__ == '__main__':
    TestSR()
