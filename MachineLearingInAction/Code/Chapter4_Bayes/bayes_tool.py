#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/27
# @Author  : Wenhao Shan
# @Dsc     : bayes learning tool

from utils.errors import ActionError


class BayesTool(object):
    """
    朴素贝叶斯工具类
    """

    @staticmethod
    def create_vocab_list(data_set: list) -> list:
        """
        将模拟数据去重合成一个单一词汇表,
        每一个词汇代表一个特征(去重)
        >>> BayesTool.create_vocab_list([['my', 'dog'], ['my']])
        list('my', 'dog')
        :return:
        """
        vocab_set = set(list())
        for document in data_set:
            vocab_set = vocab_set | set(document)  # |: 集合合并操作
        return list(vocab_set)

    @staticmethod
    def set_word_2_vec(vocab_list: list, input_set: list):
        """
        词集模型
        检测input_set中的词汇是否属于vocab_list词汇表, 如果是则置为1, 否则为0,
        实质就是将创建vocab_list的等长向量, 然后input_set转换为vocab_list的词向量(0， 1: 表示是否出现该词汇(0: 未出现, 1: 出现))
        >>> BayesTool.set_word_2_vec(['my', 'dog'], ['dog'])
        list(0, 1)
        :param vocab_list:
        :param input_set:
        :return:
        """
        return_vec = [0] * len(vocab_list)
        for word in input_set:
            if word in vocab_list:
                return_vec[vocab_list.index(word)] = 1
            else:
                raise ActionError("the word: %s is not in my Vocabulary!" % word)
        return return_vec

    @staticmethod
    def bag_of_words_2_vec_mn(vocab_list, input_set):
        """
        BayesTool.set_word_2_vec的改进, 采用文档词袋模型
        >>> BayesTool.bag_of_words_2_vec_mn(['my', 'dog'], ['dog', 'dog'])
        list(0, 2)
        :param vocab_list:
        :param input_set:
        :return:
        """
        return_vec = [0] * len(vocab_list)
        for word in input_set:
            if word in vocab_list:
                # 与mockData().set_word_2_vec唯一的不同, 词向量中词多次出现会累加
                return_vec[vocab_list.index(word)] += 1
        return return_vec


