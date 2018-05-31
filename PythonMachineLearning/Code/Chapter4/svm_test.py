#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/29
# @Author  : Wenhao Shan
# @Dsc     : SVM test

import numpy as np
import pickle
from PythonMachineLearning.Code.Chapter4.svm_train import TrainSvm
from PythonMachineLearning.Code.Chapter4.svm import SVM, svm_predict


def load_test_data(test_file: str):
    """
    导入测试数据
    :param test_file:
    :return: data(mat): 测试样本的特征
    """
    data = list()
    f = open(test_file)
    for line in f.readlines():
        lines = line.strip().split(' ')

        # 处理测试样本中的特征
        index = 0
        tmp = list()
        for i in range(0, len(lines)):
            li = lines[i].strip().split(":")
            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while int(li[0]) - 1 > index:
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data)


def load_save_model(svm_model_file: str):
    """
    导入SVM模型
    :param svm_model_file:
    :return: svm_model: SVM模型
    """
    with open(svm_model_file, "rb") as f:
        svm_model = pickle.load(f)
    return svm_model


def get_prediction(test_data: np.mat, svm: SVM):
    """
    对样本进行预测
    :param test_data:
    :param svm:
    :return: list: 预测所属的类别
    """
    m = np.shape(test_data)[0]
    prediction = list()
    for i in range(m):
        # 对每一个样本得到预测值
        predict = svm_predict(svm, test_data[i, :])
        # 得到最终的预测类别
        prediction.append(str(np.sign(predict)[0, 0]))
    return prediction


def save_prediction(result_file: str, preciction: list):
    """
    保存预测的结果
    :param result_file:
    :param preciction:
    :return:
    """
    f = open(result_file, "w")
    f.write("\n".join(preciction))
    f.close()


def TestSVM():
    """
    SVM测试
    :return:
    """
    # 1、导入测试数据
    print("--------- 1.load data ---------")
    test_data = load_test_data("svm_test_data")
    # 2、导入SVM模型
    print("--------- 2.load model ----------")
    svm_model = load_save_model("model_file")
    # 3、得到预测值
    print("--------- 3.get prediction ---------")
    prediction = get_prediction(test_data, svm_model)
    # 4、保存最终的预测值
    print("--------- 4.save result ----------")
    save_prediction("result", prediction)


if __name__ == '__main__':
    TestSVM()
