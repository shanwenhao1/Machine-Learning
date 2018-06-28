#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/28
# @Author  : Wenhao Shan
# @Dsc     : SVM function tools

import numpy as np
import pickle


class SVM:
    def __init__(self, data_set: np.mat, labels: np.mat, c: float, toler: float, kernel_option: tuple):
        self.train_x = data_set  # 训练特征
        self.train_y = labels   # 训练标签
        self.c = c  # 惩罚参数
        self.toler = toler  # 迭代的终止条件之一
        self.n_samples = np.shape(data_set)[0]  # 训练样本的个数
        self.alphas = np.mat(np.zeros((self.n_samples, 1)))  # 拉格朗日乘子
        self.b = 0
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))  # 保存E的缓存, E为误差
        self.kernel_opt = kernel_option  # 选用的核函数及其参数(高斯核函数)
        self.kernel_mat = calc_kernel(self.train_x, self.kernel_opt)    # 核函数的输出


def calc_kernel(train_x: np.mat, kernel_option: tuple):
    """
    计算核函数矩阵
    :param train_x: 训练样本的特征值
    :param kernel_option: 核函数的类型以及参数
    :return: (mat) 样本的核函数的值
    """
    m = np.shape(train_x)[0]    # 样本的个数
    kernel_matrix = np.mat(np.zeros((m, m)))    # 初始化样本之间的核函数值
    for i in range(m):
        kernel_matrix[:, i] = cal_kernel_value(train_x, train_x[i, :], kernel_option)
    return kernel_matrix


def cal_kernel_value(train_x: np.mat, train_x_i: np.mat, kernel_option):
    """
    样本之间核函数的值
    :param train_x: 训练样本
    :param train_x_i: 第i个训练样本
    :param kernel_option: 核函数的类型以及参数
    :return: (mat) 样本之间的核函数的值
    """
    kernel_type = kernel_option[0]  # 核函数的类型, 分为rbf和其他
    m = np.shape(train_x)[0]    # 样本的个数

    kernel_value = np.mat(np.zeros((m, 1)))

    # 若没有指定核函数的类型, 则默认不使用核函数
    if kernel_type == "rbf":
        sigma = kernel_option[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(m):
            diff = train_x[i, :] - train_x_i
            # 高斯核函数, P68
            kernel_value[i] = np.exp(diff * diff.T / (-2.0 * sigma ** 2))
    else:  # 不使用核函数
        kernel_value = train_x * train_x_i.T
    return kernel_value


def cal_error(svm: SVM, alpha_k: int):
    """
    误差值的计算
    :param svm:
    :param alpha_k: 选择出的变量
    :return: (float) 误差值
    """
    # w^T + b
    output_k = float(np.multiply(svm.alphas, svm.train_y).T * svm.kernel_mat[:, alpha_k] + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


def select_second_sample_j(svm: SVM, alpha_i: int, error_i: float):
    """
    选择第二个样本,  选择的标准是使其|E1-E2|改变最大
    :param svm: SVM 模型
    :param alpha_i: 选择出的第一个变量
    :param error_i: E_i
    :return: alpha_j(int) 选择出的第二个变量
            error_j(float)  E_j
    """
    # 标记为已被优化
    svm.error_tmp[alpha_i] = [1, error_i]
    candidate_alpha_list = np.nonzero(svm.error_tmp[:, 0].A)[0]   # np.nonzero返回数组中不为0的下标,
    # .A将矩阵转化为array数组类型

    max_step = 0
    alpha_j = 0
    error_j = 0

    if len(candidate_alpha_list) > 1:
        for alpha_k in candidate_alpha_list:
            if alpha_k == alpha_i:
                continue
            error_k = cal_error(svm, alpha_k)
            if abs(error_k - error_j) > max_step:
                max_step = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    else:   # 随机选择
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(np.random.uniform(0, svm.n_samples))
        error_j = cal_error(svm, alpha_j)
    return alpha_j, error_j


def choose_and_update(svm: SVM, alpha_i: int):
    """
    判断和选择两个alpha进行更新， SMO算法最核心的部分
    :param svm: SVM模型
    :param alpha_i: 选择出的第一个变量
    :return:
    """
    error_i = cal_error(svm, alpha_i)   # 计算第一个样本的E_i

    # 判断选择出的第一个变量是否违反了KKT条件
    if (svm.train_y[alpha_i] * error_i < - svm.toler) and (svm.alphas[alpha_i] < svm.c) or \
            (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

        # 1、 选择第二个变量, 选择标准是让|E1−E2|有足够大的变化
        alpha_j, error_j = select_second_sample_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # 2、 计算上下界
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:    # y_i != y_j时
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])   # L = max(0, α_j - α_i)
            H = min(svm.c, svm.c + svm.alphas[alpha_j] - svm.alphas[alpha_i])   # H = min(C, C + α_j - α_i)
        else:   # 当y_i = y_j时
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.c)   # L = max(0, α_j + α_i - C)
            H = min(svm.c, svm.alphas[alpha_j] + svm.alphas[alpha_i])   # H = min(C, α_j + α_i)
        if L == H:
            return 0

        # 3、 计算eta, -(K_11 + K_22 - 2K_12)
        eta = 2.0 * svm.kernel_mat[alpha_i, alpha_j] - svm.kernel_mat[alpha_i, alpha_i] - \
            svm.kernel_mat[alpha_j, alpha_j]
        if eta >= 0:
            return 0

        # 4、 更新alpha_j, α_j = α_i + y_j * (E1-E2)/(K_11 + K_22 - 2K_12)
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta

        # 5、 确定最终的alpha_j
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        # 6、 判断是否结束
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            update_error_tmp(svm, alpha_j)
            return 0

        # 7、 更新alpha_i, α_i_new = α_i_old + y_i * y_j * (α_j_old - α_j_new)
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] * (alpha_j_old - svm.alphas[alpha_j])

        # 8、 更新b_1和b_2
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                                                    * svm.kernel_mat[alpha_i, alpha_i] \
                             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
                                                    * svm.kernel_mat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                                                    * svm.kernel_mat[alpha_i, alpha_j] \
                             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
                                                    * svm.kernel_mat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.c):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.c):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        # 9、更新error
        update_error_tmp(svm, alpha_j)
        update_error_tmp(svm, alpha_i)

        return 1
    else:
        return 0


def update_error_tmp(svm: SVM, alpha_k: int):
    """
    重新计算误差值
    :param svm: SVM模型
    :param alpha_k: 选出的变量
    :return: 对应误差值
    """
    error = cal_error(svm, alpha_k)
    svm.error_tmp[alpha_k] = [1, error]


def SVMTraning(train_x: np.mat, train_y: np.mat, c: float, toler: float, max_iter: int,
               kernel_option: tuple=('rbf', 0.431029)):
    """
    SVM的训练, 优先进行非边界样本遍历, 将不满足KKT条件的样本进行调整, 直到整个训练集都满足KKT条件为止
    :param train_x: 训练数据的特征
    :param train_y: 训练数据的标签
    :param c: 惩罚系数
    :param toler: 迭代的终止条件之一
    :param max_iter: 最大迭代次数
    :param kernel_option: 核函数的类型及其参数
    :return: SVM模型
    """
    # 1、 初始化SVM分类器
    svm = SVM(train_x, train_y, c, toler, kernel_option)

    # 2、 开始训练
    entire_set = True
    alpha_pairs_changed = 0
    iteration = 0

    while (iteration < max_iter) and ((alpha_pairs_changed > 0) or entire_set):
        print("\t iteration: ", iteration)
        alpha_pairs_changed = 0

        if entire_set:
            # 对所有的样本
            for x in range(svm.n_samples):
                alpha_pairs_changed += choose_and_update(svm, x)
        else:
            # 非边界样本
            bound_samples = [i for i in range(svm.n_samples) if svm.alphas[i, 0] > 0 and svm.alphas[i, 0] < svm.c]
            for x in bound_samples:
                alpha_pairs_changed += choose_and_update(svm, x)
        iteration += 1

        # 在所有样本和非边界样本之间交替
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True

    return svm


def svm_predict(svm: SVM, test_sample_x: np.mat):
    """
    利用SVM模型对每一个样本进行预测
    :param svm: SVM模型
    :param test_sample_x: 样本
    :return: 对样本的预测
    """
    # 1、 计算核函数矩阵
    kernel_value = cal_kernel_value(svm.train_x, test_sample_x, svm.kernel_opt)
    # 2、 计算预测值
    predict = kernel_value.T * np.multiply(svm.train_y, svm.alphas) + svm.b
    return predict


def cal_accuracy(svm: SVM, test_x: np.mat, test_y: np.mat):
    """
    计算预测的准确性
    :param svm: SVM模型
    :param test_x: 测试的特征
    :param test_y: 测试的标签
    :return: 预测的准确性
    """
    n_samples = np.shape(test_x)[0]  # 样本的个数
    correct = 0.0
    for i in range(n_samples):
        # 对每一个样本得到预测值
        predict = svm_predict(svm, test_x[i, :])
        # 判断每一个样本的预测值的符号与真实值的符号是否一致,
        # numpy array 比较https://codeday.me/bug/20170720/43394.html
        # np.sign: 大于0的返回1.0; 小于0的返回-1.0; 等于0的返回0.0
        if np.sign(predict) == np.sign(test_y[i]):
            correct += 1
    accuracy = correct / n_samples
    return accuracy


def save_svm_model(svm_model: SVM, model_file: str):
    """
    :param svm_model: SVM模型
    :param model_file: SVM模型需要保存的文件名
    """
    with open(model_file, "wb") as f:
        pickle.dump(svm_model, f)
