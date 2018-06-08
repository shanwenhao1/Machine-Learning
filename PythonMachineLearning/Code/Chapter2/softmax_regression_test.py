#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/23
# @Author  : Wenhao Shan
import numpy as np
import random
from PythonMachineLearning.Code.Chapter2.softmax_regression_train import sr_train
from PythonMachineLearning.functionUtils import LoadModel


def load_data(num: int, m: int):
    """
    导入测试数据
    :param num: 生成的测试样本的个数
    :param m:   样本的维数
    :return:        (mat)   生成的测试样本
    """
    test_data_set = np.mat(np.ones((num, m)))
    for i in range(num):
        # 随机生成[-3, 3]之间的随机数
        test_data_set[i, 1] = random.random() * 6 - 3
        # 随机生成[0, 15]之间的随机数
        test_data_set[i, 2] = random.random() * 15
    return test_data_set


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


def test_sr():
    """
    测试Softmax Regression
    :return:
    """
    sr_train()
    print("---------- 1.Load Model ------------")
    with LoadModel("weights") as model:
        w = model.load_model_mul()
    m, n = np.shape(w)
    print("---------- 2.Load Data ------------")
    test_data = load_data(4000, m)
    print("---------- 3.Prediction ------------")
    result = predict(test_data, w)
    print("---------- 4.Save Prediction ------------")
    save_result("result", result)


if __name__ == '__main__':
    test_sr()
