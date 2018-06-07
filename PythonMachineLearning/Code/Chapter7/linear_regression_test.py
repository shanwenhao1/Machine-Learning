#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/6
# @Author  : Wenhao Shan
# @Dsc     : Linear Regression test

import numpy as np
from PythonMachineLearning.functionUtils import PaintingWithList, get_list_from_mat


def load_data(file_name: str):
    """
    导入测试数据
    :param file_name:
    :return: feature(mat): 特征
    """
    f = open(file_name)
    feature = list()
    for line in f.readlines():
        feature_tmp = list()
        lines = line.strip().split("\t")
        feature_tmp.append(1)
        for i in range(len(lines)):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
    f.close()
    return np.mat(feature)


def load_model(model_file: str):
    """
    导入模型
    :param model_file:
    :return: w(mat): 权重
    """
    f = open(model_file)
    w = [float(line.strip()) for line in f.readlines()]
    f.close()
    return np.mat(w).T


def get_prediction(data: np.mat, w: np.mat):
    """
    得到预测值
    :param data: 测试数据
    :param w: 权重值
    :return:
    """
    return data * w


def save_predict(file_name: str, predict: np.mat):
    """
    保存最终的模型
    :param file_name: 需要保存的文件名
    :param predict: 对测试数据的预测值
    :return:
    """
    m = np.shape(predict)[0]
    result = [str(predict[i, 0]) for i in range(m)]
    f = open(file_name, "w")
    f.write("\n".join(result))
    f.close()


def linear_regression_test():
    """
    线性回归测试
    :return:
    """
    # 1、导入测试数据
    testData = load_data("data_test.txt")
    # 2、导入线性回归模型
    w = load_model("weights")
    # 3、得到预测结果
    predict = get_prediction(testData, w)

    x_data = get_list_from_mat(testData, 1, True)
    y_data = get_list_from_mat(predict, 0)
    with PaintingWithList(name="Linear Regression Training") as paint:
        paint.painting_no_offset(x_data, y_data)
    # 4、保存最终的结果
    save_predict("predict_result", predict)


if __name__ == '__main__':
    linear_regression_test()
