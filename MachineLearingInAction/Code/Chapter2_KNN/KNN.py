#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/5
# @Author  : Wenhao Shan
# Dsc:

import os
# 运算符模块
from numpy import *
from utils.errors import ActionError
from MachineLearingInAction import functionUtils as FTool


class KnnLearning(object):
    """
    KNN learning
    """

    def __init__(self, group=None, labels=None):
        self.group = group
        self.labels = labels
        if self.group:
            if len(self.group) != len(self.labels):
                raise ActionError("样本数据长度不一致")
            if not isinstance(self.group, ndarray):
                raise ActionError("样本数据错误")

    @staticmethod
    def classify0(inx: ndarray, data_set: ndarray, labels: list, k: int):
        """
        KNN分类, 具体步骤:
            第一步: 先将待判定数据用tile函数复制成样本数据相同的格式, 如样本数据为4 * 2的二维数组 , 则待定数据也copy成
                    4*2的二维数组(其中每个1*2数组都为待判定数据)
            第二步: 求出每个样本数据跟待判定数据的得距离, 并排序
            第三步: 按照k值选择出跟待判定数据最接近的k个样本点, 根据这k个样本点所属类型判断待判定点的类型(样本所属类型
            最多的类型)
        :param inx: 测试数据集中的单个数据(经过归一化)
        :param data_set: 数据集
        :param labels: 标签
        :param k: k值选择(用于选择最近邻居的数目)
        :return:
        """
        # -------------------------------------------距离计算
        data_set_size = data_set.shape[0]
        # 测试数据与所有训练数据的feature点的差值(供计算距离使用)
        diff_mat = tile(inx, (data_set_size, 1)) - data_set
        # 对feature差值做平方(为求距离做准备), 如[[1, -0.1]]求平方后变为[[1, 0.01]]
        sq_diff_mat = diff_mat ** 2
        # 测试点与所有训练样本的距离, .sum将矩阵的每一行向量相加, 如sum([[0,1,2],[2,1,3],axis=1) 结果为: array([3, 6])
        sq_distances = sq_diff_mat.sum(axis=1)
        # 对平方和进行开根号(此步骤可以考虑省略, 因为平方和可以直接作比较)
        distances = sq_distances ** 0.5
        # .argsort()返回排序后的索引值, 其中靠前的表示越靠近样本的点(升序排序)
        sorted_dist_index = distances.argsort()

        # 取k个靠近样本点附近的点的类别, 进行计数
        class_count = dict()
        for i in range(k):
            vote_label_i = labels[sorted_dist_index[i]]
            class_count[vote_label_i] = class_count.get(vote_label_i, 0) + 1
        # 根据计数来判定样本点所属类别
        sorted_class_count = sorted(class_count.items(), key=lambda item: item[1], reverse=True)
        return sorted_class_count[0][0]


def classify_person(file_name: str, k_number: int):
    """
    约会网站预测函数, 根据用户输入的对方信息和以往该用户对对象看法进行分类
    :return:
    """
    result_list = ["not at all", "in small doses", "in large doses"]
    percent_tats = float(input("percentage of time spent playing video games?"))
    fly_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    with FTool.LD(file_name) as ld:
        dating_data_mat, dating_labels = ld.load_to_ndarray(3, True)
    norm_mat, ranges, min_val = FTool.HM.average(dating_data_mat)
    in_arr = array([fly_miles, percent_tats, ice_cream])
    classifier_result = KnnLearning.classify0((in_arr - min_val) / ranges, norm_mat, dating_labels, k_number)
    print("You will probably like this person: ", result_list[classifier_result - 1])


def knn_run():
    """
    Run Knn
    :return:
    """
    # # 模拟数据集
    # group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # # 模拟标签
    # labels = ['A', 'A', 'B', 'B']
    # KnnLearning(group, labels).run([-1, 1], 3)
    os_path = os.getcwd()
    file_path = os_path + "/datingTestSet2.txt"
    # KnnLearning().depart_appointment(file_path, Appointment.PlayGame, Appointment.FlyMile)
    classify_person(file_path, 3)


if __name__ == '__main__':
    knn_run()
