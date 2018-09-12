#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/9
# @Author  : Wenhao Shan
# Dsc      : Random Forest learning

from math import log
import matplotlib.pyplot as plt


def create_data_set():
    """
    用来生成模拟数据
    :return:
    """
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    label = ['no surfacing', 'flippers']
    return data_set, label


def shannon_entropy(data_set: list):
    """
    shannon entropy function, return shannon entropy of  data set
    :param data_set:
    :return:
    """
    num_entries = len(data_set)
    label_counts_dict = dict()
    # 统计各个标签类的数据数量
    for feat_vec in data_set:
        cur_label = feat_vec[-1]
        if cur_label not in label_counts_dict.keys():
            label_counts_dict[cur_label] = 0
        label_counts_dict[cur_label] += 1
    shannon_ent = 0.0
    for key in label_counts_dict.keys():
        prob = float(label_counts_dict[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)       # shannon formula
    return shannon_ent


def split_data_set(data_set: list, axis: int, value):
    """
    divide data set according to sub_list[axis] == value, the sub_list is the element of data_set
    for example:
    >>> my_dat = [[1, 1, 'yes'], [1, 1, 'yes'], [0, 0, 'no']]
    >>> split_data_set(my_dat, 0, 1)
        [[1, 'yes'], [1, 'yes']]
    >>> split_data_set(my_dat, 0, 0)
        [[0, 'no']]
    :param data_set:
    :param axis:
    :param value: any type
    :return:
    """
    ret_data_set = list()
    for feature_vec in data_set:
        if feature_vec[axis] == value:
            reduced_fea_vec = feature_vec[: axis]
            reduced_fea_vec.extend(feature_vec[axis + 1:])
            ret_data_set.append(reduced_fea_vec)
    return ret_data_set


def choose_best_fea_to_split(data_set: list):
    """
    choose the best split way to divide data set
    :param data_set:
    :return:
    """
    fea_len = len(data_set[0]) - 1
    # 初始香农熵
    base_entropy = shannon_entropy(data_set)
    best_info_gain = 0.0
    best_fea = -1
    for i in range(fea_len):
        # 利用列表推导式取出data_set中所有样本第i个特征的值
        fea_list = [_data[i] for _data in data_set]
        # 利用set过滤相同的值, 剩下的即为该特征值的所有可能性
        unique_val = set(fea_list)
        new_entropy = 0.0
        for value in unique_val:                       # 利用循环计算出每个可能的香农熵值
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * shannon_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        # 计算最好的信息增益(从而筛选出最佳划分点)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_fea = i
    return best_fea


def majority_cnt(class_list: list):
    """
    Sorted by every class count and return(统计各个分类数量, 并排序返回)
    :param class_list:
    :return:
    """
    class_count = dict()
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count += 1
    class_count_sorted = sorted(
        class_count.items(),
        key=lambda _count: _count[1],
        reverse=True)
    return class_count_sorted[0][0]


def create_tree(data_set: list, labels: list):
    """
    构建树
    :param data_set:
    :param labels:
    :return:
    """
    class_list = [_data[-1] for _data in data_set]
    # 类别完全相同时停止划分树
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 当所有特征遍历完后, 返回出现次数最多的类别
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)

    # 选择最佳划分树的特征值
    best_fea_index = choose_best_fea_to_split(data_set)
    best_fea_label = labels[best_fea_index]

    # 递归构造树
    my_tree = {best_fea_label: dict()}
    # 构造树后删除当前节点(父节点)
    del(labels[best_fea_index])
    best_fea_values = [_data[best_fea_index] for _data in data_set]
    unique_val = set(best_fea_values)   # 作为划分节点的值的可能性集合, n个值则表示划分为n个叶子节点
    # 每个叶子节点递归调用继续向下构造树
    for value in unique_val:
        sub_labels = labels[:]
        my_tree[best_fea_label][value] = create_tree(
            split_data_set(data_set, best_fea_index, value), sub_labels)
    return my_tree


def classify(input_tree: dict, fea_labels: list, test_vec: list):
    """
    测试决策树划分
    :param input_tree:
    :param fea_labels:
    :param test_vec:
    :return:
    """
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = fea_labels.index(first_str)
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    if isinstance(value_of_feat, dict):
        class_label = classify(value_of_feat, fea_labels, test_vec)
    else:
        class_label = value_of_feat
    return class_label


if __name__ == '__main__':
    my_dat, labels = create_data_set()
    # print("Test func shannon_entropy: ", shannon_entropy(my_dat))
    # print("Test func split_data_set:", split_data_set(my_dat, 0, 1))
    # print("Test func choose_best_fea_to_split:", choose_best_fea_to_split(my_dat))
    # print("Test func create_tree", create_tree(my_dat, labels))
    my_tree = create_tree(my_dat, labels)
    print("Test func classify: ", classify(my_tree, labels, [1, 1]))
