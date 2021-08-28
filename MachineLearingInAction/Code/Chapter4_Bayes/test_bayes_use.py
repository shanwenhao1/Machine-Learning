#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/27
# @Author  : Wenhao Shan

import os
import re
import unittest
import feedparser
from numpy import array, random

from MachineLearingInAction.Code.Chapter4_Bayes.bayes_use import BayesEmailExample, BayesPersonalAv
from MachineLearingInAction.Code.Chapter4_Bayes.bayes_tool import BayesTool


class MockData(object):
    """
    模拟数据
    """

    @staticmethod
    def parse(big_string: str):
        # 用正则表达式来划分字符串, \W(所有非字母、数字、下划线的字符)
        list_of_tokens = re.split(r'\W+', big_string)
        # 祛除少于两个字符的字符串并小写
        return [tok.lower() for tok in list_of_tokens if len(tok) > 2]

    def load_data(self):
        """
        加载邮件样本数据
        :return:
        """
        origin_data_list = []
        doc_list = []  # 邮件文档集合
        class_list = []  # 邮件label 集合
        # 垃圾邮件样本
        spam_dir = os.path.join('email', 'email', 'spam')
        spam_file = os.listdir(spam_dir)
        # 正常邮件样本
        ham_dir = os.path.join('email', 'email', 'ham')
        ham_file = os.listdir(ham_dir)
        # 读取文件数据并按照非字母、非制表符、非数字分割文档并添加相应的邮件标记(是否是垃圾邮件: 1: 是, 0: 否)
        for i in range(len(spam_file)):
            data = open(os.path.join(spam_dir, spam_file[i])).read()
            word_list = self.parse(data)
            origin_data_list.append(data)
            doc_list.append(word_list)
            class_list.append(1)
        for i in range(len(ham_file)):
            data = open(os.path.join(ham_dir, spam_file[i])).read()
            word_list = self.parse(data)
            origin_data_list.append(data)
            doc_list.append(word_list)
            class_list.append(0)

        # 所有邮件词汇词集
        vocab_list = BayesTool.create_vocab_list(doc_list)  # create vocabulary
        # 选取50份样本中的10份邮件样本作为测试集
        training_set = list(range(50))
        test_set = []  # create test set
        for i in range(10):
            rand_index = int(random.uniform(0, len(training_set)))
            test_set.append(training_set[rand_index])
            # 训练集中删除被选为测试集的样本
            del (training_set[rand_index])
        # 训练集中的样本转换为词向量
        train_mat = []
        train_classes = []
        for doc_index in training_set:  # train the classifier (get probs) train0
            train_mat.append(BayesTool.bag_of_words_2_vec_mn(vocab_list, doc_list[doc_index]))
            train_classes.append(class_list[doc_index])
        # 测试集中的样本转换为词向量
        test_origin_data = []
        test_mat = []
        test_classes = []
        for doc_index in test_set:
            test_origin_data.append(origin_data_list[doc_index])
            test_mat.append(BayesTool.bag_of_words_2_vec_mn(vocab_list, doc_list[doc_index]))
            test_classes.append(class_list[doc_index])
        return array(train_mat), array(train_classes), array(test_mat), array(test_classes), test_origin_data

    @staticmethod
    def calc_most_freq(vocab_list, full_text):
        """
        统计词汇表中的每个词出现的次数并排序(从高到低), 返回前30个单词
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

    def personal_av_load_data(self, feed1, feed0):
        """
        加载个人广告区域数据
        :param feed1:
        :param feed0:
        :return:
        """
        doc_list = []
        class_list = []
        full_text = []
        min_len = min(len(feed1['entries']), len(feed0['entries']))
        for i in range(min_len):
            word_list = self.parse(feed1['entries'][i]['summary'])
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(1)  # NY is class 1
            word_list = self.parse(feed0['entries'][i]['summary'])
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(0)
        vocab_list = BayesTool.create_vocab_list(doc_list)  # create vocabulary
        top30_words = self.calc_most_freq(vocab_list, full_text)  # remove top 30 words
        for pair_w in top30_words:
            if pair_w[0] in vocab_list: vocab_list.remove(pair_w[0])
        training_set = range(2 * min_len)
        test_set = []  # create test set
        for i in range(20):
            rand_index = int(random.uniform(0, len(training_set)))
            test_set.append(training_set[rand_index])
            del (training_set[rand_index])
        train_mat = [];
        train_classes = []
        for docIndex in training_set:  # train the classifier (get probs) trainNB0
            train_mat.append(BayesTool.bag_of_words_2_vec_mn(vocab_list, doc_list[docIndex]))
            train_classes.append(class_list[docIndex])
        return vocab_list


class TestBayesUse(unittest.TestCase):
    """
    垃圾邮件过滤测试
    """

    def test_spam_predict(self):
        """
        垃圾邮件过滤测试
        :return:
        """
        train_mat, train_classes, test_mat, test_classes, test_origin_data = MockData().load_data()
        BayesEmailExample(train_mat, train_classes, test_mat, test_classes).spam_predict(test_origin_data)

    def test_personal_av(self):
        """
        个人广告获取区域倾向测试
        :return:
        """
        # 书中的源无法访问, 可登陆 https://newyork.craigslist.org/ 寻找其他源代替
        # ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
        # sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
        # -----------------------------------------------------------------------
        # ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
        # sf = feedparser.parse('http://rss.cnn.com/rss/cnn_topstories.rss')
        # 为了方便这里我保存到了本地
        ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
        sf = feedparser.parse('http://rss.cnn.com/rss/cnn_topstories.rss')
        vocab_list = MockData().personal_av_load_data(ny, sf)
        BayesPersonalAv().get_top_words(ny, sf)
