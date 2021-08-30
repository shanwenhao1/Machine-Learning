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


class BayesBase(object):
    """
    朴素贝叶斯用例基础类
    """
    def __init__(self, train_mat: array, train_classes: array, test_mat: array, test_classes: array):
        self.train_mat = train_mat
        self.train_classes = train_classes
        self.test_mat = test_mat
        self.test_classes = test_classes

    def predict(self, test_origin_data: list):
        """
        训练及测试
            根据训练集训练数据, 然后根据练出的结果预测测试集中的数据属于哪个分类。
            将预测与实际结果匹配得出错误率。
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
        return p0_v, p1_v, p_spam


class BayesEmailExample(BayesBase):
    """
    朴素贝叶斯邮件过滤样例
    """
    def __init__(self, train_mat: array, train_classes: array, test_mat: array, test_classes: array):
        super(BayesEmailExample, self).__init__(train_mat, train_classes, test_mat, test_classes)

    def spam_predict(self, test_origin_data: list):
        """
        训练及测试数据
            根据训练集训练数据, 然后根据练出的结果预测测试集中的数据属于哪个分类。
            将预测与实际结果匹配得出错误率。
        :param test_origin_data: 无其他作用, 只是self.test_mat对应的原始数据, 这里用来输出预测错的邮件, 直观显示
        :return:
        """
        self.predict(test_origin_data)


class BayesPersonalAv(BayesBase):
    """
    个人广告获取区域倾向示例
    """
    def __init__(self, train_mat: array, train_classes: array, test_mat: array, test_classes: array):
        super(BayesPersonalAv, self).__init__(train_mat, train_classes, test_mat, test_classes)

    def local_words(self, vocab_list, test_origin_data: list):
        """
        训练及测试数据
            根据训练集训练数据, 然后根据练出的结果预测测试集中的数据属于哪个分类。
            将预测与实际结果匹配得出错误率。
        :param vocab_list:
        :param test_origin_data:
        :return:
        """
        p0_v, p1_v, p_spam = self.predict(test_origin_data)

        # 获取不同class中的word 及对应概率
        top_ny = []
        # print(sorted(p0_v, reverse=True))
        # print(sorted(p1_v, reverse=True))
        top_sf = []
        for i in range(len(p0_v)):
            # tuple方式存储
            top_sf.append((vocab_list[i], p0_v[i]))
            top_ny.append((vocab_list[i], p1_v[i]))
        sorted_sf = sorted(top_sf, key=lambda pair: pair[1], reverse=True)
        print("Top word of sf: SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
        # 截取前10个top词汇(根据概率排序)
        i = 0
        for item in sorted_sf:
            if i >= 10:
                break
            print(item[0])
            i += 1
        sorted_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)
        print("Top word of ny: NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
        i = 0
        for item in sorted_ny:
            if i >= 10:
                break
            print(item[0])
            i += 1


