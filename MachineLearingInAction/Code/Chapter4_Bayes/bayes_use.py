#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/26
# @Author  : Wenhao Shan
# @Dsc     : Naive Bayes use. 电子邮件垃圾过滤样例

import re
import os
import operator
from numpy import array, random

from MachineLearingInAction.Code.Chapter4_Bayes.bayes_tool import BayesTool
from MachineLearingInAction.Code.Chapter4_Bayes.bayes import BayesLearning


class BayesEmailExample(object):
    """
    朴素贝叶斯邮件过滤样例
    """
    def __init__(self, train_mat: array, train_classes: array, test_mat: array, test_classes: array):
        self.train_mat = train_mat
        self.train_classes = train_classes
        self.test_mat = test_mat
        self.test_classes = test_classes

    def spam_predict(self, test_origin_data: list):
        """
        训练及测试数据
        :param test_origin_data: 无其他作用, 只是self.test_mat对应的原始数据, 这里用来输出预测错的邮件, 直观显示
        :return:
        """
        # 训练
        p0_v, p1_v, p_spam = BayesLearning.train0(self.train_mat, self.train_classes)
        # 测试
        error_count = 0
        for i in range(len(self.test_mat)):  # classify the remaining items
            if BayesLearning.classify(self.test_mat[i], p0_v, p1_v, p_spam) != self.test_classes[i]:
                error_count += 1
                print("--------- classification error ---------\n", test_origin_data[i])
                print("------------------\n")
        print('the error rate is: %f %%' % (float(error_count) / len(self.test_classes) * 100))


class BayesPersonalAv(object):
    """
    个人广告获取区域倾向示例
    """
    def __init__(self, train_mat: array, train_classes: array, test_mat: array, test_classes: array):
        self.train_mat = train_mat
        self.train_classes = train_classes
        self.test_mat = test_mat
        self.test_classes = test_classes

    @staticmethod
    def calc_most_freq(vocab_list, full_text):
        """
        计算出现频率
        :param vocab_list:
        :param full_text:
        :return:
        """
        import operator
        freq_dict = {}
        for token in vocab_list:
            freq_dict[token] = full_text.count(token)
        sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_freq[:30]

    def local_words(self):
        p0_v, p1_v, p_spam = BayesLearning.train0(self.train_mat, self.train_classes)
        error_count = 0
        for docIndex in self.test_mat:  # classify the remaining items
            word_vector = BayesTool.bag_of_words_2_vec_mn(vocab_list, doc_list[docIndex])
            if BayesLearning.classify(array(word_vector), p0_v, p1_v, p_spam) != class_list[docIndex]:
                error_count += 1
        print('the error rate is: %f %%' % (float(error_count) / len(self.test_classes) * 100))
        return vocab_list, p0_v, p1_v

    def get_top_words(self, ny, sf):
        """

        :param ny:
        :param sf:
        :return:
        """
        vocab_list, p0_v, p1_v = self.local_words(ny, sf)
        top_ny = [];
        top_sf = []
        for i in range(len(p0_v)):
            if p0_v[i] > -6.0: top_sf.append((vocab_list[i], p0_v[i]))
            if p1_v[i] > -6.0: top_ny.append((vocab_list[i], p1_v[i]))
        sorted_sf = sorted(top_sf, key=lambda pair: pair[1], reverse=True)
        print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
        for item in sorted_sf:
            print(item[0])
        sorted_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)
        print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
        for item in sorted_ny:
            print(item[0])

