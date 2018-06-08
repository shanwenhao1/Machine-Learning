#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/8
# @Author  : Wenhao Shan
# @Dsc     : cart_train training

import numpy as np
import pickle
from PythonMachineLearning import functionUtils as FTool


class Node:
    """
    树节点的类
    """

    def __init__(self, fea: int=-1, value=None, result=None, right=None, left=None):
        self.fea = fea  # 用于切分数据集的属性的列索引值
        self.value = value  # 设置划分的值
        self.result = result    # 存储叶节点的值
        self.right = right  # 右子树
        self.left = left    # 左子树


def leaf(data_set: list):
    """
    计算叶节点的值
    :param data_set:
    :return: 均值
    """
    data = np.mat(data_set)
    # np.mean()当axis=None, 对所有m * n个数求平均值
    return np.mean(data[:, -1])


def err_cnt(data_set: list):
    """
    回归树的划分指标
    :param data_set: 训练数据
    :return: m * s^2(float): 总方差
    """
    data = np.mat(data_set)
    # np.var计算方差方差
    return np.var(data[:, -1]) * np.shape(data)[0]


def predict(sample: list, tree: Node):
    """
    对每一个样本sample进行预测
    :param sample: 样本
    :param tree: 训练好的CART回归树模型
    :return: (float): 预测值
    """
    # 1、只是树根
    if tree.result:
        return tree.result
    # 2、有左右子树
    else:
        val_sample = sample[tree.fea]   # fea处的值
        branch = None
        # 2.1、选择右子树
        if val_sample >= tree.value:
            branch = tree.right
        # 2.2、选择左子树
        else:
            branch = tree.left
        return predict(sample, branch)


def cal_error(data: list, tree: Node):
    """
    评估CART回归树模型
    :param data:
    :param tree: 训练好的CART回归树模型
    :return: 均方误差
    """
    m = len(data)   # 样本的个数
    n = len(data[0]) - 1    # 样本中特征的个数
    err = 0.0
    for i in range(m):
        tmp = [data[i][j] for j in range(n)]
        pre = predict(tmp, tree)    # 对样本计算其预测值
        # 计算残差, 利用残差平方和来评估正确率
        err += (data[i][-1] - pre) * (data[i][-1] - pre)
    return err / m


def split_tree(data: list, fea: float, value: float):
    """
    根据特征fea中的值value将数据集data划分成左右子树
    :param data: 训练样本
    :param fea: 需要划分的特征index
    :param value: 指定的划分的值
    :return: (set_1, set_2)(tuple): 左右子树的聚合
    """
    set_1 = list()  # 右子树的集合
    set_2 = list()  # 左子树的集合
    for x in data:
        if x[fea] >= value:
            set_1.append(x)
        else:
            set_2.append(x)
    return set_1, set_2


def build_tree(data: list, min_sample: int, min_err: float):
    """
    构建树
    :param data: 训练样本
    :param min_sample: 叶子节点中最少的样本数
    :param min_err: 最小的err
    :return: (Node): 树的根节点
    """
    # 构建决策树, 函数返回该决策树的根节点

    # 如果节点中的样本个数小于或等于指定的最小的样本数则该节点不再划分
    if len(data) <= min_sample:
        return Node(result=leaf(data))

    # 1、初始化
    best_err = err_cnt(data)    # 计算当前节点的error值
    best_criteria = None    # 存储最佳切分属性以及最佳切分点
    best_set = None  # 存储切分后的两个数据集

    # 2、开始构建CART回归树
    feature_num = len(data[0]) - 1
    for fea in range(0, feature_num):
        feature_values = {sample[fea]: 1 for sample in data}

        for value in feature_values.keys():
            # 2.1、尝试划分
            set_1, set_2 = split_tree(data, fea, value)
            if len(set_1) < 2 or len(set_2) < 2:
                continue
            # 2.2、计算划分后的error值
            now_err = err_cnt(set_1) + err_cnt(set_2)
            # 2.3、更新最优划分
            if now_err < best_err and len(set_1) > 0 and len(set_2) > 0:
                best_err = now_err
                best_criteria = (fea, value)
                best_set = set_1, set_2

    # 3、判断划分是否结束
    if best_err > min_err:
        right = build_tree(best_set[0], min_sample, min_err)
        left = build_tree(best_set[1], min_sample, min_err)
        return Node(fea=best_criteria[0], value=best_criteria[1], right=right, left=left)
    else:
        return Node(result=leaf(data))  # 返回当前的类别标签作为最终的类别标签


def save_model(regression_tree: Node, result_file: str):
    """
    将训练好的CART回归树模型保存到本地
    :param regression_tree: 回归树模型
    :param result_file:
    :return:
    """
    with open(result_file, "wb") as f:
        pickle.dump(regression_tree, f)


def cart_train():
    """
    cart_train training
    :return:
    """
    # 1、导入训练数据
    print("----------- 1、load data -------------")
    data, label, _ = FTool.LoadData(file_name="sine.txt").load_data(
        need_label_length=True, need_list=True, feature_end=0)
    feature = [_data[0] for _data in data]
    with FTool.PaintingWithList(name="cart_train Train") as paint:
        paint.painting_simple_list(feature, label)
    # 2、构建CART树
    print("----------- 2、build cart_train ------------")
    regression_tree = build_tree(data, 30, 0.3)
    # 3、评估CART树
    print("----------- 3、cal err -------------")
    err = cal_error(data, regression_tree)
    print("\t--------- err : ", err)
    # 4、保存最终的CART模型
    print("----------- 4、save result -----------")
    save_model(regression_tree, "regression_tree")


if __name__ == '__main__':
    cart_train()
