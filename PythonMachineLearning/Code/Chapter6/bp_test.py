#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/1
# @Author  : Wenhao Shan
# @Dsc     : BP Neural Network test

import numpy as np
from PythonMachineLearning.functionUtils import PaintingWithLabel
from PythonMachineLearning.Code.Chapter6 import bp_train as BPTrain


def generate_data():
    """
    在[-4.5, 4.5]之间随机生成20000组点
    :return:
    """
    # 1、随机生成数据点
    data = np.mat(np.zeros((20000, 2)))
    m = np.shape(data)[0]
    x = np.mat(np.random.rand(20000, 2))
    for i in range(m):
        data[i, 0] = x[i, 0] * 9 - 4.5
        data[i, 1] = x[i, 1] * 9 - 4.5
    # 2、将数据点保存到文件"test_data"中
    f = open("test_data", "w")
    m, n = np.shape(data)
    for i in range(m):
        tmp = [str(data[i, j]) for j in range(n)]
        f.write("\t".join(tmp) + "\n")
    f.close()


def load_data(file_name: str):
    """
    导入数据
    :param file_name:
    :return: feature_data(mat): 特征
    """
    f = open(file_name)
    feature_data = list()
    for line in f.readlines():
        lines = line.strip().split("\t")
        feature_tmp = [float(lines[i]) for i in range(len(lines))]
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data)


def load_model(file_w0: str, file_w1: str, file_b0: str, file_b1: str):
    """
    导入模型
    :param file_w0:
    :param file_w1:
    :param file_b0:
    :param file_b1:
    :return:
    """
    def get_model(file_name: str):
        f = open(file_name)
        model = list()
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = [float(x.strip()) for x in lines]
            model.append(model_tmp)
        f.close()
        return np.mat(model)

    # 1、导入输入层到隐含层之间的权重
    w0 = get_model(file_w0)

    # 2、导入隐含层到输出层之间的权重
    w1 = get_model(file_w1)

    # 1、导入输入层到隐含层之间的偏置
    b0 = get_model(file_b0)

    # 1、导入隐含层到输出层之间的偏置
    b1 = get_model(file_b1)
    return w0, w1, b0, b1


def save_predict(file_name: str, pre: np.mat):
    """
    保存最终的预测结果
    :param file_name:
    :param pre:
    :return:
    """
    f = open(file_name, "w")
    m = np.shape(pre)[0]
    result = [str(pre[i, 0]) for i in range(m)]
    f.write("\n".join(result))
    f.close()


def BPTest():
    """
    BP test
    :return:
    """
    # BPTrain.BPTrain()

    generate_data()
    # 1、导入测试数据
    print("--------- 1.load data ------------")
    data_test = load_data("test_data")
    # 2、导入BP神经网络模型
    print("--------- 2.load model ------------")
    w0, w1, b0, b1 = load_model("weight_w0", "weight_w1", "weight_b0", "weight_b1")
    # 3、得到最终的预测值
    print("--------- 3.get prediction ------------")
    result = BPTrain.get_predict(data_test, w0, w1, b0, b1)
    # 4、保存最终的预测结果
    print("--------- 4.save result ------------")
    pre = np.argmax(result, axis=1)
    with PaintingWithLabel(name="BP Test") as paint:
        paint.painting_with_no_offset(data_test, pre)
    save_predict("result", pre)


if __name__ == '__main__':
    BPTest()
