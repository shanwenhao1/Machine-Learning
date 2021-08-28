#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/15
# @Author  : Wenhao Shan
# @Dsc     : Naive Bayes Learning

from numpy import mat, matrix, log, ones, array
from utils.errors import ActionError


class BayesLearning(object):
    """
    Bayes Learning, 以斑点狗留言板为例
    """
    def __init__(self):
        pass

    @staticmethod
    def train0(train_matrix: array, train_category: array):
        """
        训练函数
        :param train_matrix: 文档矩阵(各个样本组成, 每个样本已经转换为词向量), example:
        :param train_category: 文档矩阵对应类别标签所组成的向量
        :return:
        """
        # 文档数量
        num_train_docs = len(train_matrix)
        # 每个文档包含的词数量
        num_words = len(train_matrix[0])
        # 样本中任意文档属于侮辱性文档的概率
        p_abusive = sum(train_category)/float(num_train_docs)
        # 初始化概率
        # 非侮辱性词汇出现概率初始化, ones(5) => array([1., 1., 1., 1., 1.])
        '''之所以初始化为1, 是因为贝叶斯分类器对文档进行分类时, 要计算多个概率的乘积以获得文档属于某个类别的概率, 即
        p(W0|1)p(W1|1)p(W2|1)...为防止因其中一个概率为0的情况导致乘积也为0. 因此统一将所有词出现初始化为1, 
        并将分母初始化为2
        '''
        p0_num = ones(num_words)
        # 侮辱性词汇出现概率初始化
        p1_num = ones(num_words)
        p0_denom = 2.0
        p1_denom = 2.0
        # 对每篇训练文档
        for i in range(num_train_docs):
            # 如果该文档为侮辱性文档, 则计数(注意为矩阵相加, 因此每个侮辱性词汇出现的次数都会计数)
            if train_category[i] == 1:
                p1_num += train_matrix[i]
                # 统计总的词汇数量
                p1_denom += sum(train_matrix[i])
            else:
                p0_num += train_matrix[i]
                p0_denom += sum(train_matrix[i])
        # 侮辱性词汇1, 2, 3...出现的概率
        '''
        这里在概率计算加上log的原因是由于最后统计文档概率相乘时, 由于每个概率都是小数, 太多小数相乘会导致下溢出(乘积为0). 
        因此对乘积使用自然对数(ln(a*b) = ln(a) + ln(b)避免了下溢出问题, 并且由于f(x)和ln(f(x))的曲线相同, 因此不影响求极值)
        '''
        # p1_vec_t: 所有词在侮辱性文档中出现的概率的词向量(添加了log)
        p1_vec_t = log(p1_num/p1_denom)
        # p0_vec_t: 所有词在非侮辱性文档中出现的概率的词向量(添加了log)
        p0_vec_t = log(p0_num/p0_denom)
        return p0_vec_t, p1_vec_t, p_abusive

    @staticmethod
    def classify(vec2_classify: array, p0_vec, p1_vec, p_class1):
        """
        bayes预测函数
        :param vec2_classify: 待预测的词向量(需根据训练样本的所有词组成的list生成)
        :param p0_vec: 所有词在非侮辱性文档中出现的概率的词向量(添加了log)
        :param p1_vec: 所有词在侮辱性文档中出现的概率的词向量(添加了log)
        :param p_class1: 类别概率(这里以侮辱性文档概率为例: 即公式里的p(c))
        :return:
        """
        # 因为使用了log, 这里加法就是乘法(参考: ln(x1*x2) = ln(x1) + ln(x2)), 因为log(f(x))跟f(x)的函数曲线一致
        # 当f(x1) > f(x2)时, log(f(x1)) > log(f(x2)). 因此可使用log(f(x))代替f(x)进行比较
        '''
        这里sum(vec2_classify * p1_vec)相当于公式中的p(x|c)/p(x)将x展开成一个个独立的特征. 这里的
        p(x|c)/p(x) = (p(x1|c) * p(x2|c) * ...)
        '''
        p1 = sum(vec2_classify * p1_vec) + log(p_class1)
        # 因为这里只有两种情况, 因此非侮辱性概率= 1-侮辱性概率
        p0 = sum(vec2_classify * p0_vec) + log(1.0 - p_class1)
        if p1 > p0:
            return 1
        else:
            return 0

