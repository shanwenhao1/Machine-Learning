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

    @staticmethod
    def get_train_and_test_data(vocab_list, doc_list, class_list, training_set_num, test_set_num, origin_data_list):
        """
        获取训练集和测试集数据
        :param vocab_list:
        :param doc_list:
        :param class_list:
        :param training_set_num:
        :param test_set_num:
        :param origin_data_list:
        :return:
        """
        training_set = list(range(training_set_num))
        test_set = []  # create test set
        for i in range(test_set_num):
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
        return self.get_train_and_test_data(vocab_list, doc_list, class_list, 50, 10, origin_data_list)

    @staticmethod
    def calc_most_freq(vocab_list, full_text):
        """
        统计词汇表中的每个词出现的次数并排序(从高到低), 返回前30个单词
        :param vocab_list: 待排序抽取前30个词的词汇表
        :param full_text: 所有样本文档(用于统计词汇出现次数)
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
        origin_data_list = []
        doc_list = []
        class_list = []
        full_text = []
        min_len = min(len(feed1['entries']), len(feed0['entries']))
        # 抽取取两个样本中相同的样本数, 这里为60(min_len)
        for i in range(min_len):
            origin_data_list.append(feed1['entries'][i]['summary'])
            word_list = self.parse(feed1['entries'][i]['summary'])
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(1)    # NY is class 1

            origin_data_list.append(feed0['entries'][i]['summary'])
            word_list = self.parse(feed0['entries'][i]['summary'])
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(0)    # SY is class 0
        vocab_list = BayesTool.create_vocab_list(doc_list)  # create vocabulary
        # 抽取前30个高频词汇
        top30_words = self.calc_most_freq(vocab_list, full_text)  # remove top 30 words
        # 移除抽取出的30个高频词汇
        '''
            移除这些词汇的目的是因为: 这些高频词汇占用了所用词相当大的比例, 会大幅增加预测的错误率. 除了移除高频词汇外
            另外一种解决方法是不仅移除高频词汇还建立一个停用词表用于移除结构上的辅助词.
                https://www.ranks.nl/stopwords
        '''
        for pair_w in top30_words:
            if pair_w[0] in vocab_list:
                vocab_list.remove(pair_w[0])
        # 抽取120个样本中的20个样本作为测试样本
        train_mat, train_classes, test_mat, test_classes, test_origin_data = self.get_train_and_test_data(vocab_list,
                                                                                                          doc_list,
                                                                                                          class_list,
                                                                                                          2 * min_len,
                                                                                                          20,
                                                                                                          origin_data_list)
        return vocab_list, train_mat, train_classes, test_mat, test_classes, test_origin_data


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
            这里错误率远高于垃圾邮件中的错误率是因为这里关注的是单词概率而不是实际分类, 可通过更改
            >>> MockData.calc_most_freq
            中移除的单词数目观察错误率变化
        :return:
        """
        # 书中的源无法访问, 可登陆 https://newyork.craigslist.org/ 寻找其他源代替
        # ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
        # sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
        # -----------------------------------------------------------------------
        # ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
        # sf = feedparser.parse('http://rss.cnn.com/rss/cnn_topstories.rss')
        # 为了方便这里我保存到了本地
        ny = feedparser.parse('image_of_the_day.rss')
        sf = feedparser.parse('rss_cnn_topstories.rss')
        vocab_list, train_mat, train_classes, test_mat, test_classes, test_origin_data = \
            MockData().personal_av_load_data(ny, sf)
        # 训练及测试数据
        BayesPersonalAv(train_mat, train_classes, test_mat, test_classes).local_words(vocab_list, test_origin_data)
