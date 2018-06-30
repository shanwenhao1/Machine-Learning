#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/14
# @Author  : Wenhao Shan
# @Dsc     : Matrix Factorization-based Recommend

import numpy as np
from PythonMachineLearning import functionUtils as FTool
from PythonMachineLearning.Code.Chapter14.user_based_recommend import top_k


def grad_ascent(data: np.mat, k: int, alpha: float, beta: float, max_cycles: int):
    """
    利用梯度下降法对矩阵进行分解
    :param data: 用户商品矩阵
    :param k: 分解矩阵的参数
    :param alpha: 学习率
    :param beta: 正则化参数
    :param max_cycles: 最大迭代次数
    :return: p,q(mat): 分解后的矩阵
    """
    m, n = np.shape(data)
    # 1、初始化p和q
    p = np.mat(np.random.random((m, k)))
    q = np.mat(np.random.random((k, n)))

    # 2、开始训练
    for step in range(max_cycles):
        # 2.1、根据负梯度更新向量
        for i in range(m):
            for j in range(n):
                if data[i, j] <= 0:  # "-"的情况则跳过
                    continue
                error = data[i, j]
                # 根据矩阵乘法定义的data[i, j] = p第i行, q第j列对应乘积的和, 即预测值data'[i, j]
                for r in range(k):
                    error -= p[i, r] * q[r, j]
                for r in range(k):
                    # 梯度上升, 对应GD更新公式
                    p[i, r] += alpha * (2 * error * q[r, j] - beta * p[i, r])
                    q[r, j] += alpha * (2 * error * p[i, r] - beta * q[r, j])

        # 2.2、计算损失函数值
        loss = 0.0
        for i in range(m):
            for j in range(n):
                if data[i, j] <= 0:
                    continue
                error = 0.0
                for r in range(k):
                    error += p[i, r] * q[r, j]
                # 计算损失函数
                loss = (data[i, j] - error) * (data[i, j] - error)
                for r in range(k):
                    loss += beta * (p[i, r] * p[i, r] + q[r, j] * q[r, j]) / 2
        # 2.3、损失函数值满足条件则退出
        if loss < 0.001:
            break
        if step % 1000 == 0:
            print("iter: ", step, "loss: ", loss)
    return p, q


def prediction(data_matrix: np.mat, p: np.mat, q: np.mat, user: int):
    """
    为用户user未互动的项打分
    :param data_matrix: 原始用户商品矩阵
    :param p: 分解后的矩阵p
    :param q: 分解后的矩阵q
    :param user: 用户的id
    :return: predict(list): 推荐列表
    """
    n = np.shape(data_matrix)[1]
    predict = {j: (p[user, ] * q[:, j])[0, 0] for j in range(n) if data_matrix[user, j] == 0}
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)


def mf_run():
    """
    Matrix Factorization-based Recommend run
    :return:
    """
    # 1、导入用户商品矩阵
    print("----------- 1、load data -----------")
    data_matrix = FTool.LoadData(file_name="data.txt").load_data_with_none()
    # 2、利用梯度下降法对矩阵进行分解
    print("----------- 2、training -----------")
    p, q = grad_ascent(data_matrix, 3, 0.0002, 0.02, 5000)  # 由于样本过少, 迭代次数增加min loss也不会继续收敛
    # 3、保存分解后的结果
    print("----------- 3、save decompose -----------")
    with FTool.SaveModel(file_name="p_result") as save_file:
        save_file.save_model_mul(p)
    with FTool.SaveModel(file_name="q_result") as save_file:
        save_file.save_model_mul(q)
    # 4、预测
    print("----------- 4、prediction -----------")
    predict = prediction(data_matrix, p, q, 0)
    # 进行Top-K推荐
    print("----------- 5、top_k recommendation ------------")
    top_recommend = top_k(predict, 2)
    print(top_recommend)
    print(p * q)


if __name__ == '__main__':
    mf_run()
