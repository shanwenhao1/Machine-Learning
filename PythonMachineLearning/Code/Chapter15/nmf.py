#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/14
# @Author  : Wenhao Shan
# @Dsc     : Non-Negative-Matrix Factorization-based

import numpy as np


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

        a = w_matrix.T * v_data
        b = w_matrix.T * w_matrix * h_matrix