#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/30
# @Author  : Wenhao Shan
# @Dsc     : The Random Forest Test

import numpy as np
import pickle
from PythonMachineLearning.Code.Chapter5 import random_forests_train as RFTrain


def load_data(file_name: str):
    """
    导入待分类的数据集
    :param file_name:
    :return:
    """
    f = open(file_name)
    test_data = list()
    for line in f.readlines():
        lines = line.strip().split("\t")
        tmp = [float(x) for x in lines]
        tmp.append(0)   # 保存初始的label
        test_data.append(tmp)
    f.close()
    return test_data


def load_model(result_file: str, feature_file: str):
    """
    导入随机森林模型和每一个分类树中选择的特征
    :param result_file:
    :param feature_file:
    :return: trees_result(list): 随机森林模型
              trees_feature(list): 每一棵树选择的特征
    """
    # 1、导入选择的特征
    trees_feature = list()
    f_fea = open(feature_file)
    for line in f_fea.readlines():
        lines = line.strip().split("\t")
        tmp = [int(x) for x in lines]
        trees_feature.append(tmp)
    f_fea.close()

    # 2、导入随机森林模型
    with open(result_file, "rb") as f:
        trees_result = pickle.load(f)
    return trees_result, trees_feature


def save_result(data_test: list, prediction: list, result_file: str):
    """
    保存最终的预测结果
    :param data_test:
    :param prediction:
    :param result_file:
    :return:
    """
    m = len(prediction)
    n = len(data_test[0])

    f_result = open(result_file, "w")
    for i in range(m):
        tmp = [str(data_test[i][j]) for j in range(n - 1)]
        tmp.append(str(prediction[i]))
        f_result.writelines("\t".join(tmp) + "\n")
    f_result.close()


def TestRandomForest():
    """
    测试RF分类
    :return:
    """
    # 1、导入测试数据集
    print("--------- 1、load test data --------")
    data_test = load_data("test_data.txt")
    # 2、导入随机森林模型
    print("--------- 2、load random forest model ----------")
    trees_result, trees_feature = load_model("result_file", "feature_file")
    # 3、预测
    print("--------- 3、get prediction -----------")
    prediction = RFTrain.get_predict(trees_result, trees_feature, data_test)
    # 4、保存最终的预测结果
    print("--------- 4、save result -----------")
    save_result(data_test, prediction, "final_result")


if __name__ == '__main__':
    TestRandomForest()
