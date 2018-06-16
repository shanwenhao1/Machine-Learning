#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/23
# @Author  : Wenhao Shan

import numpy as np
from random import normalvariate    # 正态分布
from PythonMachineLearning.functionUtils import sig, PaintingWithMat


def load_data_set(data: str):
    """
    导入训练数据
    :param data:    训练数据
    :return:       data_mat(list)   特征
                    label_mat(list)  标签
    """
    data_mat = list()
    # data_mat_show只是用来画图所用
    data_mat_show = list()
    label_mat = list()
    fr = open(data)
    for line in fr.readlines():
        lines = line.strip().split("\t")
        line_arr = [float(lines[i]) for i in range(len(lines) - 1)]
        arr_show = [0]
        arr_show.extend(line_arr)

        data_mat.append(line_arr)
        data_mat_show.append(arr_show)

        label_mat.append(float(lines[-1]) * 2 - 1)   # 转换成{-1, 1}
    fr.close()
    with PaintingWithMat(name="FM Point") as paint:
        paint.painting_with_offset(np.mat(data_mat_show), np.mat(label_mat).T)
    return data_mat, label_mat


def initialize_v(n: int, k: int):
    """
    初始化交叉项
    :param n:   特征的个数
    :param k:   FM模型的超参数
    :return:    (mat)   交叉项的系数权重
    """
    v = np.mat(np.zeros((n, k)))

    for i in range(n):
        for j in range(k):
            # 利用正态分布生成每一个权重
            v[i, j] = normalvariate(0, 0.2)
    return v


def stocGradAscent(dataMatrix: np.mat, classLabels: np.mat, k: int, max_iter: int, alpha: float):
    """
    利用随机梯度下降法训练FM模型
    :param dataMatrix:  特征
    :param classLabels: 标签
    :param k:   v的维数
    :param max_iter:    最大的迭代次数
    :param alpha:   学习率
    :return:                w0(float)  偏置项
                             w(mat)     一次项权重
                             v(mat)     交叉项的权重
    """
    m, n = np.shape(dataMatrix)
    # 1、初始化参数
    w = np.zeros((n, 1))    # 其中n是特征的个数
    w0 = 0  # 偏置项
    v = initialize_v(n, k)  # 初始化v

    # 2、训练
    for it in range(max_iter):
        for x in range(m):      # 随机优化, 对每一个样本而言的
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)    # multiply对应元素相乘
            # 完成交叉项
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2  # 对应模型公式

            p = w0 + dataMatrix[x] * w + interaction    # 计算预测的输出
            loss = sig(classLabels[x] * p[0, 0]) - 1

            w0 -= alpha * loss * classLabels[x]
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] -= alpha * loss * classLabels[x] * dataMatrix[x, i]

                    for j in range(k):
                        v[i, j] -= alpha * loss * classLabels[x] * (dataMatrix[x, i] * inter_1[0, j] - v[i, j]
                                                                    * dataMatrix[x, i] * dataMatrix[x, i])
        # 计算损失函数的值
        if it % 1000 == 0:
            print("\t---------------- iter: ", it, " , cost: ",
                  get_cost(getPrediction(np.mat(dataMatrix), w0, w, v), classLabels))

    # 3、返回最终的FM模型的参数
    return w0, w, v


def get_cost(preidct: list, classLabels: list):
    """
    计算预测准确性
    :param preidct: 预测值
    :param classLabels: 标签
    :return:            (float) 计算损失函数的值
    """
    m = len(preidct)
    error = 0.0
    for i in range(m):
        error -= np.log(sig(preidct[i] * classLabels[i]))
    return error


def getPrediction(dataMatrix: np.mat, w0: float, w: float, v: float):
    """
    得到预测值
    :param dataMatrix:  特征
    :param w0:  一次项权重
    :param w:   常数项权重
    :param v:   交叉项权重
    :return:
    """
    m = np.shape(dataMatrix)[0]
    result = []
    for x in range(m):

        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
            np.multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
        p = w0 + dataMatrix[x] * w + interaction  # 计算预测的输出
        pre = sig(p[0, 0])
        result.append(pre)
    return result


def getAccuracy(predict: list, classLabels: list):
    """
    计算预测准确性
    :param predict: 预测值
    :param classLabels: 标签
    :return: 错误率
    """
    m = len(predict)
    allItem = 0
    error = 0
    for i in range(m):
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue
    return float(error) / allItem


def save_model(file_name: str, w0: float, w: np.mat, v: np.mat):
    """
    保存训练好的FM模型
    :param file_name: 文件名
    :param w0: 偏置项
    :param w:  一次项的权重
    :param v:  交叉项的权重
    :return:
    """
    f = open(file_name, "w")
    # 1、保存w0
    f.write(str(w0) + "\n")
    # 2、保存一次项的权重
    w_array = list()
    m = np.shape(w)[0]
    for i in range(m):
        w_array.append(str(w[i, 0]))
    f.write("\t".join(w_array) + "\n")
    # 3、保存交叉项的权重
    m1, n1 = np.shape(v)
    for i in range(m1):
        v_tmp = [str(v[i, j]) for j in range(n1)]
        f.write("\t".join(v_tmp) + "\n")
    f.close()


def TrainFM():
    """
    训练FM模型
    :return:
    """
    # 1、导入训练数据
    print("---------- 1.Load Data ---------")
    dataTrain, labelTrain = load_data_set("data.txt")
    print("---------- 2.Learning ---------")
    # 2、利用随机梯度训练FM模型
    w0, w, v = stocGradAscent(np.mat(dataTrain), labelTrain, 3, 10000, 0.01)
    predict_result = getPrediction(np.mat(dataTrain), w0, w, v)  # 得到训练的准确性
    print("----------Training Accuracy: %f" % (1 - getAccuracy(predict_result, labelTrain)))
    print("---------- 3.Save Result ---------")
    # 3、保存训练好的FM模型
    save_model("weights", w0, w, v)


if __name__ == '__main__':
    TrainFM()
