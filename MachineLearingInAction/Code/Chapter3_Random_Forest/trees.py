#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/9
# @Author  : Wenhao Shan
# Dsc      : Random Forest learning

from math import log
from numpy import ndarray


def shannon_entropy(data_set: ndarray):
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
