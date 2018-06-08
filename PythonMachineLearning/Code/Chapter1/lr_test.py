#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/21
# @Author  : Wenhao Shan

import numpy as np
from PythonMachineLearning.Code.Chapter1 import lr_train
from PythonMachineLearning import functionUtils as FTool


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
    with FTool.LoadModel("weights") as model:
        w = model.load_model_mul()
    n = np.shape(w)[1]

    print("------------------------2. Load Data-----------------------")
    test_data = FTool.LoadData(file_name="test_data", feature_type="float").load_data_with_limit(number=n, offset=1)

    print("------------------------3. Get Prediction-----------------------")
    h = predict(test_data, w)

    # TODO painting
    print("------------------------4. Save Prediction-----------------------")
    save_result("result", h)


if __name__ == '__main__':
    TestOfLR()
