#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/13
# @Author  : Wenhao Shan
# Dsc      : Test of tree(include create, use) and painting tree

import os
import pickle
import unittest
from MachineLearingInAction import functionUtils as FTool
from MachineLearingInAction.Code.Chapter3_Random_Forest.trees import create_tree, classify
from MachineLearingInAction.Code.Chapter3_Random_Forest.tree_plot import PaintingTree


class TestClass(unittest.TestCase):
    """
    决策树test
    """
    @classmethod
    def create_data_set(cls):
        """
        用来生成模拟数据
        :return:
        """
        # 样本数据
        data_set = ["young", "hyper", "no", "normal"]
        # label列表
        label = ['age', 'prescript', 'astigmatic', 'tearRate']
        return data_set, label

    @staticmethod
    def store_tree(input_tree: dict, filename: str):
        """
        使用pickle模块储存决策树
        :param input_tree:
        :param filename:
        :return:
        """
        fw = open(filename, 'wb')
        pickle.dump(input_tree, fw)
        fw.close()

    @staticmethod
    def grab_tree(filename: str):
        """
        加载决策树
        :param filename:
        :return:
        """
        fr = open(filename, 'rb')
        return pickle.load(fr)

    def test_create_tree(self):
        """
        测试创建决策树
        :return:
        """
        file_path = os.path.join(os.getcwd(), "lenses.txt")
        with FTool.LD(file_path) as ld:
            lenses = ld.load_data_list()
            # 年龄, 材质硬度 散光, 流泪频率共四个特征. 第五个为结论
            lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
            lenses_tree = create_tree(lenses, lenses_labels)
            # 画出决策树
            print(lenses_tree)
            PaintingTree().create_plot(lenses_tree)
            self.store_tree(lenses_tree, 'classifierStorage.txt')

    def test_classify(self):
        """
        测试使用决策树进行分类
        :return:
        """
        tree = self.grab_tree('classifierStorage.txt')
        test_dat, test_labels = self.create_data_set()
        label = classify(tree, test_labels, test_dat)
        self.assertEqual("soft", label, "Prediction result error")
