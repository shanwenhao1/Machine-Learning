#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/8
# @Author  : Wenhao Shan

from numpy import *

from MachineLearingInAction.k_nearest_neighbor.KNN import KnnLearning


def datingClassTest(file_name: str, k_number: int):
    """
    注意测试分类算法是否正确, 样本集和测试集必须随机从总的样本集里面取, 由于本次测试的数据并没有按照特定目的排序,
    因此我们只是顺序挑取测试集(前10%)
    :return:
    """
    hoRatio = 0.10
    dating_mat, dating_labels = KnnLearning.file_matrix(file_name)
    norm_mat, ranges, min_val = KnnLearning.auto_norm(dating_mat)
    m = norm_mat.shape[0]
    # 校验数据的数量, 一般为10%, 90%样本数据用来训练分类器
    num_test_vec = int(m * hoRatio)
    error_count = 0
    for i in range(num_test_vec):
        # norm_mat[num_test_vec: m, :] 中num_test_vec: m是截取ndarray中第num_test_vec个后的数据
        classifier_result = KnnLearning._classify0(norm_mat[i, :], norm_mat[num_test_vec: m, :],
                                                   dating_labels[num_test_vec: m], k_number)
        print("The classifier came back with {0}, the real answer is: {1}".format(classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1
    error_rate = error_count / num_test_vec
    print("The total error rate is {}".format(error_rate))


if __name__ == '__main__':
    file_path = os.getcwd() + "/datingTestSet2.txt"
    datingClassTest(file_path, 3)
