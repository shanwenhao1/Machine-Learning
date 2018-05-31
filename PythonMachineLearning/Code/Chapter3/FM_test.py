#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/25
# @Author  : Wenhao Shan

import numpy as np
from PythonMachineLearning.Code.Chapter3.FM_train import getPrediction


def loadDataSet(data: str):
    """
    导入测试数据集
    :param data:
    :return: (list) 特征
    """
    dataMat = list()
    fr = open(data)
    for line in fr.readlines():
        lines = line.strip().split("\t")

        lineArr = [float(lines[i]) for i in range(len(lines))]
        dataMat.append(lineArr)

    fr.close()
    return dataMat


def loadModel(model_file: str):
    """
    导入FM模型
    :param model_file:
    :return:    (mat)      w0
                 (mat)w.T
                 (mat)      v
    """
    f = open(model_file)
    line_index = 0
    w0 = 0.0
    w = list()
    v = list()
    for line in f.readlines():
        lines = line.strip().split("\t")
        if line_index == 0:
            w0 = float(lines[0].strip())
        elif line_index == 1:
            for x in lines:
                w.append(float(x.strip()))
        else:
            v_tmp = [float(x.strip()) for x in lines]
            v.append(v_tmp)
        line_index += 1
    f.close()
    return w0, np.mat(w).T, np.mat(v)


def save_result(file_name: str, result: np.mat):
    """
    保存最终的预测结果
    :param file_name: 保存的文件名
    :param result: 预测结果
    :return:
    """
    f = open(file_name, "w")
    f.write("\n".join(str(x) for x in result))
    f.close()


def TestFM():
    """
    测试FM模型
    :return:
    """
    # 1、导入测试数据
    dataTest = loadDataSet("test_data.txt")
    # 2、导入FM模型
    w0, w, v = loadModel("weights")
    # 3、预测
    result = getPrediction(dataTest, w0, w, v)
    # 4、保存最终的预测结果
    save_result("predict_result", result)


if __name__ == '__main__':
    TestFM()
