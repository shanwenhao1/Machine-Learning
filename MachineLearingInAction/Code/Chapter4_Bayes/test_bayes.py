#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/25
# @Author  : Wenhao Shan

import os
import unittest
from numpy import array

from utils.errors import ActionError
from MachineLearingInAction import functionUtils as FTool
from MachineLearingInAction.Code.Chapter4_Bayes.bayes import BayesLearning
from MachineLearingInAction.Code.Chapter4_Bayes.bayes_tool import BayesTool


class mockData(object):
    """
    模拟数据
    """
    def __init__(self):
        # 模拟词条切分后的文档
        self.posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        # 人工标注词条性质
        self.class_vec = array([0, 1, 0, 1, 0, 1])  # 1: abusive, 0: not abusive

    def mock_vocab_list(self):
        """
        模拟数据生成单一词汇表
        :return:
        """
        return BayesTool.create_vocab_list(self.posting_list)

    def get_mock_data(self):
        """
        获取mock data
        :return: 词汇出现的次数矩阵
        """
        my_vocab_list = self.mock_vocab_list()
        '''
        my_vocab_list: 所有文档的词汇汇总成的词汇向量
        ['quit', 'my', 'ate', 'buying', 'posting', 'I', 'flea', 'him', 'mr', 'worthless', 'dog', 'stupid', 'so', 
        'licks', 'park', 'steak', 'maybe', 'not', 'has', 'cute', 'love', 'problems', 'help', 'garbage', 'to', 'is', 
        'dalmation', 'stop', 'how', 'please', 'food', 'take']
        '''
        train_mat = []
        # 模拟数据(0, 1矩阵)
        for post_doc in self.posting_list:
            train_mat.append(BayesTool.set_word_2_vec(my_vocab_list, post_doc))
        '''
        train_mat: 文档样本self.posting_list中各个文档中词汇(相较于my_vocab_list)出现的次数组成的矩阵
        [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1], 
        [0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        '''
        return array(train_mat)


class TestBayes(unittest.TestCase):
    """
    朴素贝叶斯算法测试
    """
    def test_train(self):
        # 样本数据
        train_data = mockData().get_mock_data()
        list_classes = mockData().class_vec
        # 训练
        p0v, p1v, pab = BayesLearning().train0(train_data, list_classes)
        # 测试数据
        test_entry = ['love', 'my', 'dalmation']
        # 测试数据转换成向量
        this_doc = array(BayesTool.set_word_2_vec(mockData().mock_vocab_list(), test_entry))
        result = BayesLearning().classify(this_doc, p0v, p1v, pab)
        self.assertEqual(0, result, "bayes predicted error, not abusive word")
        # 测试数据
        test_entry_2 = ['stupid', 'garbage']
        # 测试数据转换成向量
        this_doc_2 = array(BayesTool.set_word_2_vec(mockData().mock_vocab_list(), test_entry_2))
        result_2 = BayesLearning().classify(this_doc_2, p0v, p1v, pab)
        self.assertEqual(1, result_2, "bayes predicted error, abusive word")
