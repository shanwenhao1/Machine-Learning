#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/28
# @Author  : Wenhao Shan
# @Dsc     : The Random Forest Training

import pickle
import random
import numpy as np
from math import log
from PythonMachineLearning.functionUtils import PaintingPicture
from PythonMachineLearning.Code.Chapter5 import tree as TreeTool


def load_data(file_name: str):
    """
    导入数据
    :param file_name:
    :return: data_train(list): 训练数据
    """
    data_train = list()
    f = open(file_name)
    for line in f.readlines():
        lines = line.strip().split("\t")
        data_tmp = [float(x) for x in lines]
        data_train.append(data_tmp)
    f.close()
    with PaintingPicture(name="Random Forest Train Data ") as paint:
        paint.painting_simple_index(data_train)
    return data_train


def random_forest_training(data_train: list, trees_num: int):
    """
    构建随机森林
    :param data_train: 训练数据
    :param trees_num: 分类树的个数
    :return: trees_result(list): 每一棵树的最好划分
              trees_feature(list): 每一颗树中对原始特征的选择
    """
    trees_result = list()
    trees_feature = list()
    n = np.shape(data_train)[1]     # 样本的维数
    if n > 2:
        k = int(log(n - 1, 2) + 1)  # 设置特征的个数
    else:
        k = 1

    # 开始构建每一棵树
    for i in range(trees_num):
        # 1、随机选择m个样本, k个特征
        data_samples, feature = choose_samples(data_train, k)
        # 2、构建每一颗分类树
        tree = TreeTool.build_tree(data_samples)
        # 3、保存训练好的分类树
        trees_result.append(tree)
        # 4、保存好该分类树使用到的特征
        trees_feature.append(feature)
    return trees_result, trees_feature


def choose_samples(data: list, k: int):
    """
    从样本中随机选择样本及其特征
    :param data: 原始数据集
    :param k: 选择特征的个数
    :return: data_samples(list): 被选择出来的样本
              feature(list): 被选择的特征index
    """
    m, n = np.shape(data)   # 样本的个数和样本特征的个数
    # 1、选择出k个特征的index
    feature = [random.randint(0, n - 2) for j in range(k)]
    # 2、选择出m个样本的index, 有放回的抽取, 样本可重复
    index = [random.randint(0, m - 1) for i in range(m)]
    # 3、从data中选择m个样本的k个特征, 组成数据集data_samples
    data_samples = list()
    for i in range(m):
        data_tmp = [data[index[i]][fea] for fea in feature]
        data_tmp.append(data[index[i]][-1])
        data_samples.append(data_tmp)
    return data_samples, feature


def get_predict(trees_result: list, trees_feature: list, data_train: list):
    """
    利用训练好的随机森林模型对样本进行预测
    :param trees_result: 训练好的随机森林模型
    :param trees_feature: 每一颗分类树选择的特征
    :param data_train: 训练样本
    :return: final_predict(list): 对样本预测的结果
    """
    m_tree = len(trees_result)
    m = np.shape(data_train)[0]

    result = list()
    for i in range(m_tree):
        clf = trees_result[i]   # 第几棵分类树
        feature = trees_feature[i]  # 对应树的随机选择的特征
        data = split_data(data_train, feature)  # 数据采样, 只取出决策树所需的特征值
        # .keys()返回迭代器, 加上list转换为列表
        # 对样本集中的每一个样本进行预测
        result_i = [list(TreeTool.predict(data[j][0:-1], clf).keys())[0] for j in range(m)]
        result.append(result_i)
    final_predict = np.sum(result, axis=0)
    return final_predict


def cal_correct_rate(data_train: list, final_predict: list):
    """
    计算模型的预测准确性
    :param data_train: 训练样本
    :param final_predict: 预测结果
    :return: (float): 准确性
    """
    # 所有数据类型
    all_point_type = [index[-1] for index in data_train]
    # set去重
    all_point_type = list(set(all_point_type))
    m = len(final_predict)
    corr = 0.0
    data_type = list()  # 预测的分类结果, 只是为了画图所用
    for i in range(m):
        # 比较预测结果与原始样本中的标签, 若两者同号则表示预测正确
        if data_train[i][-1] * final_predict[i] > 0:
            data_type.append(data_train[i][-1])
            corr += 1
        else:
            # 为了方便显示随便挑选一个错误的分类
            data_type.append(all_point_type[0] if data_train[i][-1] != all_point_type[0] else all_point_type[1])
    with PaintingPicture(fig_support=211, name="Random Forest Train Data") as paint:
        paint.painting_after_train(data_train, data_type)
    return corr / m


def save_model(trees_result: list, trees_feature: list, result_file: str, feature_file: str):
    """
    保存最终的模型
    :param trees_result: 训练好的随机森林模型
    :param trees_feature: 每一棵决策树选择的特征
    :param result_file: 模型保存的文件
    :param feature_file: 特征保存的文件
    :return:
    """
    # 1、保存选择的特征
    m = len(trees_feature)
    f_fea = open(feature_file, "w")
    for i in range(m):
        fea_tmp = [str(x) for x in trees_feature[i]]
        f_fea.writelines("\t".join(fea_tmp) + "\n")
    f_fea.close()

    # 2、保存最终的随机森林模型
    with open(result_file, "wb") as f:
        pickle.dump(trees_result, f)


def split_data(data_train: list, feature: list):
    """
    选择特征(按照feature中的特征从原始数据集选出指定的特征)
    :param data_train: 训练数据集
    :param feature: 要选择的特征
    :return: data(list): 选择出来的数据集
    """
    m = np.shape(data_train)[0]
    data = list()

    for i in range(m):
        data_x_tmp = [data_train[i][x] for x in feature]
        data_x_tmp.append(data_train[i][-1])
        data.append(data_x_tmp)
    return data


def RandomForestTrain():
    """
    随机森林训练
    :return:
    """
    # 1、导入数据
    print("----------- 1、load data -----------")
    data_train = load_data("data.txt")
    # 2、训练random_forest模型
    print("----------- 2、random forest training ------------")
    trees_result, trees_feature = random_forest_training(data_train, 50)
    # 3、得到训练的准确性
    print("------------ 3、get prediction correct rate ------------")
    result = get_predict(trees_result, trees_feature, data_train)
    corr_rate = cal_correct_rate(data_train, result)
    print("\t------correct rate: ", corr_rate)
    # 4、保存最终的随机森林模型
    print("------------ 4、save model -------------")
    save_model(trees_result, trees_feature, "result_file", "feature_file")


if __name__ == '__main__':
    RandomForestTrain()
