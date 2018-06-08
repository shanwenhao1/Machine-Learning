#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/8
# @Author  : Wenhao Shan
# @Dsc     : cart_train test

import numpy as np
import random
import pickle
from PythonMachineLearning.Code.Chapter9.cart_train import Node, predict
from PythonMachineLearning import functionUtils as FTool


def load_data():
    """
    随机生成样本数据
    :return: (list): 随机生成的测试样本
    """
    data_test = list()
    for i in range(4000):
        tmp = list()
        tmp.append(random.random())     # 随机生成[0, 1]之间的样本
        data_test.append(tmp)
    return data_test


def load_model(tree_file: str):
    """
    导入训练好的CART回归树模型
    :param tree_file:
    :return: regression_tree(cart_train.Node): CART回归树
    """
    with open(tree_file, "rb") as f:
        regression_tree = pickle.load(f)
    return regression_tree


def get_prediction(data_test: list, regression_tree: Node):
    """
    对测试样本进行预测
    :param data_test: 需要预测的样本
    :param regression_tree: 训练好的回归树模型
    :return: result(list): 预测结果
    """
    result = [predict(x, regression_tree) for x in data_test]
    return result


def save_result(data_test: list, result: list, prediction_file: str):
    """
    保存最终的预测结果
    :param data_test: 需要预测的数据集
    :param result: 预测的结果
    :param prediction_file: 保存结果的文件
    :return:
    """
    f = open(prediction_file, "w")
    for i in range(len(result)):
        a = str(data_test[i][0]) + "\t" + str(result[i]) + "\n"
        f.write(a)
    f.close()


def cart_test():
    """
    cart_train 测试
    :return:
    """
    # 1、导入待计算的数据
    print("--------- 1、load data ----------")
    data_test = load_data()
    # 2、导入回归树模型
    print("--------- 2、load regression tree ---------")
    regression_tree = load_model("regression_tree")
    # 3、进行预测
    print("--------- 3、get prediction -----------")
    prediction = get_prediction(data_test, regression_tree)
    x_data = [_data[0] for _data in data_test]
    y_data = prediction
    with FTool.PaintingWithList(name="CART Test") as paint:
        paint.painting_simple_list(x_data, y_data)
    # 4、保存预测的结果
    print("--------- 4、save result ----------")
    save_result(data_test, prediction, "prediction")


if __name__ == '__main__':
    cart_test()
