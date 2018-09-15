#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/15
# @Author  : Wenhao Shan
# @Dsc     : Naive Bayes Learning

from utils.errors import ActionError


def mock_data_set():
    """
    mock test data set
    :return:
    """
    # 模拟词条切分后的文档
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]          # 1: abusive, 0: not abusive
    return posting_list, class_vec


def create_vocab_list(data_set: list):
    """
    将模拟数据去重合成一个单一词汇表,
    每一个词汇代表一个特征
    :return:
    """
    vocab_set = set(list())
    for document in data_set:
        vocab_set = vocab_set | set(document)       # |: 集合合并操作
    return list(vocab_set)


def set_word_2_vec(vocab_list: list, input_set: list):
    """
    检测input_set中的词汇是否属于vocab_list词汇表, 如果是则置为1, 否则为0,
    实质就是将导入数据转成bool类型的特征列表
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


if __name__ == '__main__':
    list_post, list_class = mock_data_set()
    my_vocab_list = create_vocab_list(list_post)
    word_2_vec = set_word_2_vec(my_vocab_list, list_post[0])
    print(word_2_vec)
