#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/8
# @Author  : Wenhao Shan
import os
from MachineLearingInAction import functionUtils as FTool
from MachineLearingInAction.Code.Chapter2_KNN.KNN import KnnLearning


def dating_class_test(file_name: str, k_number: int):
    """
    注意测试分类算法是否正确, 样本集和测试集必须随机从总的样本集里面取, 由于本次测试的数据并没有按照特定目的排序,
    因此我们只是顺序挑取测试集(前10%)
    :param file_name: 文件名
    :param k_number: 截取的k个邻居
    :return:
    """
    ho_ratio = 0.10
    with FTool.LD(file_name) as ld:
        dating_mat, dating_labels = ld.load_to_ndarray(3, True)
    with FTool.Painting3D(name="Personal hobby with love") as plt:
        plt.paint(dating_mat, dating_labels)
    norm_mat, ranges, min_val = FTool.HM.average(dating_mat)
    m = norm_mat.shape[0]
    # 校验数据的数量, 一般为10%, 90%样本数据用来训练分类器
    num_test_vec = int(m * ho_ratio)
    error_count = 0
    for i in range(num_test_vec):
        # norm_mat[num_test_vec: m, :] 中num_test_vec: m是截取ndarray中第num_test_vec个后的数据
        classifier_result = KnnLearning.classify0(norm_mat[i, :], norm_mat[num_test_vec: m, :],
                                                  dating_labels[num_test_vec: m], k_number)
        print("The classifier came back with {0}, the real answer is: {1}".format(classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1
    error_rate = error_count / num_test_vec
    print("The total error rate is {}".format(error_rate))


if __name__ == '__main__':
    file_path = os.getcwd() + "/datingTestSet2.txt"
    dating_class_test(file_path, 3)
