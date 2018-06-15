#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/15
# @Author  : Wenhao Shan
# @Dsc     : Non-Negative-Matrix Factorization-based

import numpy as np
from PythonMachineLearning import functionUtils as FTool
from PythonMachineLearning.Code.Chapter15.mf import prediction, top_k


def train(v_data: np.mat, r: int, max_cycles: int, error: float):
    """
    非负矩阵分解
    :param v_data: 评分矩阵
    :param r: 分解后矩阵的维数
    :param max_cycles: 最大的迭代次数
    :param error: 误差(退出条件)
    :return: w_matrix, h_matrix(mat): 分解后的矩阵
    """
    m, n = np.shape(v_data)
    # 1、初始化矩阵
    w_matrix = np.mat(np.random.random((m, r)))
    h_matrix = np.mat(np.random.random((r, n)))

    # 2、非负矩阵分解
    for step in range(max_cycles):
        v_pre = w_matrix * h_matrix
        err_matrix = v_data - v_pre    # 矩阵误差
        err = 0.0
        for i in range(m):
            for j in range(n):
                err += err_matrix[i, j] * err_matrix[i, j]      # 矩阵误差的平方和

        if err < error:
            break
        if step % 1000 == 0:
            print("iter: ", step, "loss: ", err)

        # 假设分解后的矩阵为P 和 Q, 原矩阵为R, 方便以下解释
        a = w_matrix.T * v_data     # P.T * R
        b = w_matrix.T * w_matrix * h_matrix    # P.T * P * Q
        for i_1 in range(r):
            for j_1 in range(n):
                if b[i_1, j_1] == 0:
                    continue
                # Q = Q * ((P.T * R) / (P.T * P * Q))
                h_matrix[i_1, j_1] = h_matrix[i_1, j_1] * a[i_1, j_1] / b[i_1, j_1]

        c = v_data * h_matrix.T  # R * Q.T
        d = w_matrix * h_matrix * h_matrix.T    # P * Q * Q.T
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    # Q = Q * ((P.T * R) / (P.T * R * Q))
                    w_matrix[i_2, j_2] = w_matrix[i_2, j_2] * c[i_2, j_2] / d[i_2, j_2]
    return w_matrix, h_matrix


def nmf_run():
    """
    非负矩阵分解 run
    :return:
    """
    # 1、导入用户商品矩阵
    print("----------- 1、load data -----------")
    data_matrix = FTool.LoadData(file_name="data.txt").load_data_with_none()
    # 2、利用梯度下降法对矩阵进行分解
    print("----------- 2、training -----------")
    w_matrix, h_matrix = train(data_matrix, 5, 10000, 1e-5)
    # 3、保存分解后的结果
    print("----------- 3、save decompose -----------")
    with FTool.SaveModel(file_name="w_matrix") as save_file:
        save_file.save_model_mul(w_matrix)
    with FTool.SaveModel(file_name="h_matrix") as save_file:
        save_file.save_model_mul(h_matrix)
    # 4、预测
    print("----------- 4、prediction -----------")
    predict = prediction(data_matrix, w_matrix, h_matrix, 0)
    # 进行Top-K推荐
    print("----------- 5、top_k recommendation ------------")
    top_recommend = top_k(predict, 2)
    print(top_recommend)
    print(w_matrix * h_matrix)


if __name__ == '__main__':
    nmf_run()
