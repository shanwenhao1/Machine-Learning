#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/6
# @Author  : Wenhao Shan
# @Dsc     :  Ridge Regression training

import numpy as np
from PythonMachineLearning.functionUtils import LoadData, PaintingWithList, SaveModel


def ridge_regression(feature: np.mat, label: np.mat, lam: float):
    """
    最小二乘法的求解方法(岭回归)
    :param feature: 特征
    :param label: 标签
    :param lam: λ系数
    :return: w(mat)回归系数
    """
    n = np.shape(feature)[1]
    w = (feature.T * feature + lam * np.mat(np.eye(n))).I * feature.T * label
    return w


def get_gradient(feature: np.mat, label: np.mat, w: np.mat, lam: float):
    """
    计算导函数的值(求梯度)
    :param feature: 特征
    :param label: 标签
    :param w:
    :param lam: λ系数
    :return: w(mat): 回归系数
    """
    err = (label - feature * w).T
    left = err * (-1) * feature
    return left.T + lam * w


def get_error(feature: np.mat, label: np.mat, w: np.mat):
    """
    矫正回归系数
    :param feature: 特征
    :param label: 标签
    :param w: 当前回归系数
    :return: w(mat): 矫正后的回归系数
    """
    m = np.shape(feature)[0]
    left = (label - feature * w).T * (label - feature * w)
    return (left / (2 * m))[0, 0]


def get_result(feature: np.mat, label: np.mat, w: np.mat, lam: float):
    """
    计算损失函数的值
    :param feature:
    :param label:
    :param w:
    :param lam:
    :return:
    """
    left = (label - feature * w).T * (label - feature * w)
    right = lam * w.T * w
    return (left + right) / 2


def bfgs(feature: np.mat, label: np.mat, lam: float, maxCycle: int):
    """
    利用bfgs训练Ridge Regression模型
    :param feature: 特征
    :param label: 标签
    :param lam: 正则化参数
    :param maxCycle: 最大迭代次数
    :return: w(mat): 回归系数
    """
    n = np.shape(feature)[1]
    # 1、初始化
    w0 = np.mat(np.zeros((n, 1)))
    rho = 0.55  # δ
    sigma = 0.4  # σ
    bk = np.eye(n)
    k = 1
    while k < maxCycle:
        print("\titer: ", k, "\terror: ", get_error(feature, label, w0))
        gk = get_gradient(feature, label, w0, lam)  # 计算梯度
        dk = np.mat(-np.linalg.solve(bk, gk))   # 求解bk * x = gk中的x矩阵
        m = 0
        mk = 0
        # Armijo线搜索
        while m < 20:
            newf = get_result(feature, label, (w0 + rho ** m * dk), lam)
            oldf = get_result(feature, label, w0, lam)
            if newf < oldf + sigma * (rho ** m) * (gk.T * dk)[0, 0]:
                mk = m
                break
            m += 1

        # BFGS校正
        w = w0 + rho ** mk * dk
        sk = w - w0
        yk = get_gradient(feature, label, w, lam) - gk
        if yk.T * sk > 0:
            bk = bk - (bk * sk * sk.T * bk) / (sk.T * bk * sk) + (yk * yk.T) / (yk.T * sk)
        k += 1
        w0 = w
    return w0


def lbfgs(feature: np.mat, label: np.mat, lam: float, maxCycle: int, save_m: int):
    """
    利用lbfgs训练Ridge Regression模型
    :param feature: 特征
    :param label: 标签
    :param lam: 正则化参数
    :param maxCycle: 最大迭代次数
    :param save_m: lbfgs中选择保留的个数
    :return:
    """
    n = np.shape(feature)[1]
    # 1、初始化
    w0 = np.mat(np.zeros((n, 1)))
    rho = 0.55  # δ
    sigma = 0.4  # σ

    H0 = np.eye(n)

    s = list()
    y = list()

    k = 1
    gk = get_gradient(feature, label, w0, lam)  # 3X1
    print(gk)
    dk = -H0 * gk
    # 2、迭代
    while k < maxCycle:
        print("iter: ", k, "\terror: ", get_error(feature, label, w0))
        m1 = 0
        mk = 0
        gk = get_gradient(feature, label, w0, lam)
        # 2.1、Armijo线搜索, 寻找符合条件的最下非负整数m
        while m1 < 20:
            new_f = get_result(feature, label, (w0 + rho ** m1 * dk), lam)
            old_f = get_result(feature, label, w0, lam)
            if new_f < old_f + sigma * (rho ** m1) * (gk.T * dk)[0, 0]:
                mk = m1
                break
            m1 += 1

        # 2.2、LBFGS校正
        w = w0 + rho ** mk * dk
        # 保留m个
        if k > save_m:
            s.pop(0)
            y.pop(0)

        # 保留最新的
        sk = w - w0
        qk = get_gradient(feature, label, w, lam)   # 3X1
        yk = qk - gk    # k+1导函数与k导函数的差值

        s.append(sk)
        y.append(yk)

        # two-loop
        t = len(s)
        a = list()
        for i in range(t):
            alpha = (s[t - i - 1].T * qk) / (y[t - i - 1].T * s[t - i - 1])
            qk = qk - alpha[0, 0] * y[t - i - 1]
            a.append(alpha[0, 0])
        r = H0 * qk

        for i in range(t):
            beta = (y[i].T * r) / (y[i].T * s[i])
            r += s[i] * (a[t - i - 1] - beta[0, 0])

        if yk.T * sk > 0:
            dk = -r

        k += 1
        w0 = w
    return w0


def ridge_regression_train(method: str):
    """
    岭回归模型训练
    :param method: 选择的方法
    :return:
    """
    # 1、导入数据
    print("----------1.load data ------------")
    feature, label, _ = LoadData(file_name="data.txt").load_data(offset=1, need_label_length=True, need_list=True)
    with PaintingWithList(name="Ridge Regression Training") as paint:
        paint.painting_with_offset(feature, label)
    feature = np.mat(feature)
    label = np.mat(label).T
    # 2、训练模型
    print("----------2.training ridge_regression ------------")
    if method == "bfgs":  # 选择BFGS训练模型
        w0 = bfgs(feature, label, 0.5, 1000)
    elif method == "lbfgs":  # 选择L-BFGS训练模型
        w0 = lbfgs(feature, label, 0.5, 1000, save_m=10)
    else:  # 使用最小二乘的方法
        w0 = ridge_regression(feature, label, 0.5)
    # 3、保存最终的模型
    print("----------3.save model ------------")
    with SaveModel(file_name="weights") as save_model:
        save_model.save_model_mul(w0)


if __name__ == '__main__':
    ridge_regression_train("lbfgs")
