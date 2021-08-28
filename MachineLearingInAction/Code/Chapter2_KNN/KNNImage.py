#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/8
# @Author  : Wenhao Shan
# @Dsc     : 手写识别系统

import os
from os import listdir
from numpy import zeros
from MachineLearingInAction.Code.Chapter2_KNN.KNN import KnnLearning


def img2vector(file_name: str):
    """
    图像转换为向量函数, 将32 * 32图像读取成1 * 1024的numpy数组
    (相当于拥有1024个特征的样本)
    :param file_name:
    :return:
    """
    return_vect = zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def get_dir_image_vector(dir_path: str):
    """
    转化指定目录下所有手写图像数据集
    :param dir_path:
    :return:
    """
    label_list = list()
    all_file = listdir(dir_path)
    file_num = len(all_file)
    train_mat = zeros((file_num, 1024))
    # 批量读取数字识别文件
    for i in range(file_num):
        file_name = all_file[i]
        file_head_name = file_name.split('.')[0]
        # 分类在文件名守个字符, 需要提取出来
        class_label = int(file_head_name.split('_')[0])
        label_list.append(class_label)
        # 解析目下文件名并调用img2vector将图像转换为矩阵
        train_mat[i, :] = img2vector(dir_path + '/%s' % file_name)
    return label_list, train_mat


def hand_writing_class():
    """
    手写数字识别系统测试, 由于特征数量太多， 因此KNN算法效率不高, K决策树属于K-近邻算法的优化版
    :return:
    """
    dir_name_train = "trainingDigits"
    dir_name_test = "testDigits"
    train_label, train_mat = get_dir_image_vector(dir_name_train)
    test_label, test_mat = get_dir_image_vector(dir_name_test)

    err_count = 0
    for i in range(len(test_label)):
        classifier_result = KnnLearning.classify0(
            test_mat[i, :], train_mat, train_label, 3)
        if classifier_result != test_label[i]:
            print(
                "The classifier came back with: %d, the real answer is: %d" %
                (classifier_result, test_label[i]))
            err_count += 1
    print("The Total number of errors is: %d\n Total error rate is: %f" %
          (err_count, err_count / float(len(test_label))))


if __name__ == '__main__':
    hand_writing_class()
