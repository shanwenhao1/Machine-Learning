#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/8
# @Author  : Wenhao Shan
# @Dsc     :  Ridge Regression test

import numpy as np
from PythonMachineLearning.functionUtils import LoadData, PaintingWithList, LoadModel
from PythonMachineLearning.Code.Chapter8 import ridge_regression_train as RRTrain


def get_prediction(data: np.mat, w: np.mat):
    """
    对新数据进行预测
    :param data: 测试数据
    :param w: 权重值
    :return: (mat): 最终的预测
    """
    return data * w


def save_result(file_name: str, predict: np.mat):
    """
    保存最终的结果
    :param file_name:
    :param predict: 预测结果
    :return:
    """
    m = np.shape(predict)[0]
    result = [str(predict[i, 0]) for i in range(m)]
    f = open(file_name, "w")
    f.write("\n".join(result))
    f.close()


def ridge_regression_test():
    """
    预测函数
    :return:
    """
    RRTrain.ridge_regression_train("lbfgs")
    # 1、导入测试数据
    print("----------1.load data ------------")
    test_data, _ = LoadData(file_name="data_test.txt").load_data(offset=1, feature_end=0)
    x_data = [test_data[i, 1] for i in range(np.shape(test_data)[0])]
    y_data = [test_data[i, 2] for i in range(np.shape(test_data)[0])]
    with PaintingWithList(name="Ridge Regression Test") as paint:
        paint.painting_no_offset(x_data, y_data, mul_simple=True)
    # 2、导入线性回归模型
    print("----------2.load model ------------")
    with LoadModel("weights") as model:
        w = model.load_model_mul(need_transpose=False)
    # 3、得到预测结果
    print("----------3.get prediction ------------")
    predict = get_prediction(test_data, w)
    # 4、保存最终的结果
    print("----------4.save prediction ------------")
    save_result("predict_result", predict)


if __name__ == '__main__':
    ridge_regression_test()
