#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/28
# @Author  : Wenhao Shan
# @Dsc     : SVM train

import re
import numpy as np
from PythonMachineLearning.Code.Chapter4 import svm


def load_data_lib_svm(data_file: str):
    """
    导入训练数据
    :param data_file: 训练样本文件名
    :return: data(mat): 训练样本的特征, label(mat): 训练样本的标签
    """
    data = list()
    label = list()
    f = open(data_file)
    for line in f.readlines():
        # lines = re.split(r" |\n", line.strip())
        lines = line.strip().split(' ')

        # 提取得出label
        label.append(float(lines[0]))
        # 提取出特征, 并将其放入到矩阵中
        index = 0
        tmp = list()
        for i in range(1, len(lines)):
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
    # painting_with_offset(np.mat(data), np.mat(label).T, "SVM Point")
    return np.mat(data), np.mat(label).T


def train_vm():
    """
    训练svm
    :return:
    """    # 1、导入训练数据
    print("------------ 1、load data --------------")
    data_set, labels = load_data_lib_svm("heart_scale")     # 特征和标签矩阵
    # 2、训练SVM模型
    print("------------ 2、training ---------------")
    c = 0.6     # 惩罚系数(正则项), 提高泛化能力
    toler = 0.001   # 迭代的终止条件之一
    max_iter = 5000  # 最大迭代次数
    svm_model = svm.SVMTraning(data_set, labels, c, toler, max_iter)
    # 3、计算训练的准确性
    print("------------ 3、cal accuracy --------------")
    accuracy = svm.cal_accuracy(svm_model, data_set, labels)
    print("The training accuracy is: %.3f%%" % (accuracy * 100))
    # 4、保存最终的SVM模型
    print("------------ 4、save model ----------------")
    svm.save_svm_model(svm_model, "model_file")


if __name__ == '__main__':
    train_vm()
