#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/13
# @Author  : Wenhao Shan
# Dsc      : Test of tree(include create, use) and painting tree

import os
from MachineLearingInAction import functionUtils as FTool
from MachineLearingInAction.Code.Chapter3_Random_Forest.trees import create_tree
from MachineLearingInAction.Code.Chapter3_Random_Forest.tree_plot import PaintingTree


if __name__ == '__main__':
    file_path = os.getcwd() + "\lenses.txt"
    with FTool.LD(file_path) as ld:
        lenses = ld.load_data_list()
        # 年龄, 材质硬度 散光, 流泪频率共四个特征
        lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lenses_tree = create_tree(lenses, lenses_labels)
        print(lenses_tree)
        PaintingTree().create_plot(lenses_tree)