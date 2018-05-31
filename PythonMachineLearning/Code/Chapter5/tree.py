#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/28
# @Author  : Wenhao Shan
# @Dsc     : Tree Base function tools

import numpy as np


class Node:
    """
    数的节点的类
    """

    def __init__(self, fea=-1, value=None, results=None, right: __name__ = None, left: __name__ = None):
        self.fea = fea  # 用于切分数据集的属性的列索引值
        self.value = value  # 设置划分的值
        self.results = results  # 存储叶节点所属的类别
        self.right = right  # 右子树
        self.left = left  # 左子树


def build_tree(data: list):
    """
    构建树
    :param data:
    :return: (Node): 树的根节点
    """
    # 构建决策树, 函数返回该决策树的根节点
    if len(data) == 0:
        return Node()

    # 1、计算当前的Gini指数
    currentGini = cal_gini_index(data)

    bestGain = 0.0
    bestCriteria = None  # 存储最佳切分属性以及最佳切分点
    bestSets = None  # 存储切分后的两个数据集

    feature_num = len(data[0]) - 1  # 样本中特征的个数
    # 2、找到最好的划分
    for fea in range(0, feature_num):
        # 2.1、取得fea特征处所有可能的取值
        feature_values = {sample[fea]: 1 for sample in data}  # 对每一个样本, 存储特征fea处所有可能的取值

        # 2.2、针对每一个可能的取值, 尝试将数据集划分, 并计算Gini指数
        for value in feature_values.keys():  # 遍历该属性的所有切分点
            # 2.2.1、根据fea特征中的值value将数据集划分成左右子树
            (set_1, set_2) = split_tree(data, fea, value)
            # 2.2.2、计算当前的Gini指数
            nowGini = float(len(set_1) * cal_gini_index(set_1) + len(set_2) * cal_gini_index(set_2)) / len(data)
            # 2.2.3、计算Gini指数的增加量
            gain = currentGini - nowGini
            # 2.2.4、判断此划分是否比当前的划分更好
            if gain > bestGain and len(set_1) > 0 and len(set_2) > 0:
                bestGain = gain
                bestCriteria = (fea, value)
                bestSets = (set_1, set_2)

    # 3、判断划分是否结束
    if bestGain > 0:
        right = build_tree(bestSets[0])
        left = build_tree(bestSets[1])
        return Node(fea=bestCriteria[0], value=bestCriteria[1], right=right, left=left)
    else:
        return Node(results=label_uniq_cat(data))   # 返回当前的类别标签作为最终的类别标签


def cal_gini_index(data: list):
    """
    计算给定数据集的Gini指数
    :param data: 数据集
    :return: gini(float): Gini指数
    """
    total_sample = len(data)  # 样本个数
    if len(data) == 0:
        return 0
    label_counts = label_uniq_cat(data)  # 统计数据集中不同标签的个数

    # 计算Gini指数
    gini = 0
    for label, value in label_counts.items():
        gini += pow(value, 2)  # 这里将分子分母分开是因为分母一样, 可以提升运算效率
    gini = 1 - float(gini) / pow(total_sample, 2)
    return gini


def label_uniq_cat(data: list):
    """
    统计数据集中不同的类别的标签label的个数
    :param data: 原始数据集
    :return: (dict) 样本中各个标签的数据个数
    """
    label_uniq_cnt = dict()
    for x in data:
        label = x[len(x) - 1]  # 取得每一个样本的标签label
        if label not in label_uniq_cnt:
            label_uniq_cnt[label] = 0
        else:
            label_uniq_cnt[label] += 1
    return label_uniq_cnt


def split_tree(data: object, fea: object, value: object) -> object:
    """
    根据特征fea中的值value将数据集data划分成左右子树(主要针对特征值是连续值)
    :param data: 数据集
    :param fea: 待分割特征的索引
    :param value: 待分割的特征的具体值
    :return: (set1, set2)(tuple): 分割后的左右子树
    """
    set_1 = list()
    set_2 = list()
    for x in data:
        if x[fea] >= value:
            set_1.append(x)
        else:
            set_2.append(x)
    result_tuple = (set_1, set_2)
    return result_tuple


def predict(sample: list, tree: Node):
    """
    对每一个样本sample进行预测
    :param sample: 需要预测的样本
    :param tree: 构建好的分类树
    :return: tree.results: 所属的类别
    """
    # 1、只是树根
    if tree.results != None:
        return tree.results
    # 2、有左右子树
    else:
        val_sample = sample[tree.fea]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
        return predict(sample, branch)  # 递归调用直到为叶子节点为止
