#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12
# @Author  : Wenhao Shan
# @Dsc     : Mean Shift use Gaussian-Kernel Function

import math
import numpy as np
from PythonMachineLearning import functionUtils as FTool

MIN_DISTANCE = 0.000001     # 最小误差


def load_data(file_path: str, feature_num: int=2):
    """
    导入数据
    :param file_path: 文件存储的位置
    :param feature_num: 特征的个数
    :return: (array): 特征
    """
    f = open(file_path)  # 打开文件
    data = list()
    for line in f.readlines():
        lines = line.strip().split("\t")
        if len(lines) != feature_num:   # 判断特征的个数是否正确
            continue
        data_tmp = [float(lines[i]) for i in range(feature_num)]
        data.append(data_tmp)
    f.close()
    return data


def gaussian_kernel(distance: np.mat, bandwidth: int):
    """
    高斯核函数
    :param distance: 欧式距离 (存放x_1 - x_2的欧式距离)
    :param bandwidth: 核函数的带宽
    :return: gaussian_val(mat): 高斯函数值
    """
    m = np.shape(distance)[0]   # 样本个数
    right = np.mat(np.zeros((m, 1)))    # m * 1矩阵
    for i in range(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T / (bandwidth * bandwidth))
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))
    gaussian_val = left * right
    return gaussian_val


def euclidean_dist(point_a: np.mat, point_b: np.mat):
    """
    计算欧式距离
    :param point_a: A点的坐标
    :param point_b: B点的坐标
    :return:
    """
    total = (point_a - point_b) * (point_a - point_b).T
    return math.sqrt(total)     # 欧氏距离


def shift_point(point: np.mat, points: np.array, kernel_bandwidth: int):
    """
    计算均值漂移点
    :param point: 需要计算的点
    :param points: 所有的样本点
    :param kernel_bandwidth: 核函数的带宽
    :return: point_shifted(mat): 漂移后的点
    """
    points = np.mat(points)
    m = np.shape(points)[0]     # 样本的个数
    # 计算距离
    point_distance = np.mat(np.zeros((m, 1)))
    for i in range(m):
        point_distance[i, 0] = euclidean_dist(point, points[i])

    # 计算高斯核
    point_weights = gaussian_kernel(point_distance, kernel_bandwidth)   # m * 1的矩阵

    # 计算分母
    all_sum = 0.0
    for i in range(m):
        all_sum += point_weights[i, 0]

    # 均值偏移
    point_shifted = point_weights.T * points / all_sum
    return point_shifted


def group_points(mean_shift_points: np.mat):
    """
    计算所属的类别
    :param mean_shift_points: 漂移向量
    :return: (array): 所属类别
    """
    group_assignment = list()
    m, n = np.shape(mean_shift_points)
    index = 0
    index_dict = dict()
    for i in range(m):
        item = [str(("%5.2f" % mean_shift_points[i, j])) for j in range(n)]
        item_1 = "_".join(item)
        if item_1 not in index_dict.keys():
            index_dict[item_1] = index
            index += 1

    for i in range(m):
        item = [str(("%5.2f" % mean_shift_points[i, j])) for j in range(n)]
        item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])
    return group_assignment


def train_mean_shift(points: np.array, kernel_bandwidth: int=2):
    """
    训练Mean shift 模型
    :param points: 特征数据
    :param kernel_bandwidth: 核函数的带宽
    :return: points(mat): 特征点
              mean_shift_points(mat): 均值漂移点, 即聚类中心
              group(array): 类别
    """
    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iteration = 0   # 训练的代数
    m = np.shape(mean_shift_points)[0]  # 样本的个数
    need_shift = [True] * m     # 标记是否需要漂移

    # 计算均值漂移向量
    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        iteration += 1
        print(" iteration: %s" % str(iteration))
        for i in range(0, m):
            # 判断每一个样本点是否需要计算偏移均值
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kernel_bandwidth)    # 对样本点进行漂移
            dist = euclidean_dist(p_new, p_new_start)   # 计算该点与漂移后的点之间的距离
            if dist > max_min_dist:
                max_min_dist = dist
            if dist < MIN_DISTANCE:     # 不需要移动
                need_shift[i] = False

            mean_shift_points[i] = p_new

        # 计算最终的group
        group = group_points(mean_shift_points)  # 计算所属的类别
    return np.mat(points), mean_shift_points, group


def mean_shift():
    """
    Mean Shift 执行函数
    :return:
    """
    # 导入数据集
    print("----------1.load data ------------")
    data = load_data("data", 2)
    x_data = [_data[0] for _data in data]
    y_data = [_data[1] for _data in data]
    with FTool.PaintingWithList(name="Mean Shift") as paint:
        paint.painting_simple_list(x_data, y_data)
    # 训练，h=2
    print("----------2.training ------------")
    points, shift_points, cluster = train_mean_shift(data, 2)
    # 保存所属的类别文件
    print("----------3.1.save sub ------------")
    with FTool.SaveModel(file_name="sub_1") as save_model:
        save_model.save_model_mul(np.mat(cluster))
    print("----------3.2.save center ------------")
    # 保存聚类中心
    with FTool.SaveModel(file_name="center_1") as save_model:
        save_model.save_model_mul(shift_points)
    after_x = list()
    after_y = list()
    for _position in range(len(shift_points)):
        after_x.append(shift_points[_position, 0])
        after_y.append(shift_points[_position, 1])
    with FTool.PaintingWithList(name="Mean Shift Result") as paint:
        paint.painting_list_with_label(after_x, after_y, cluster)


if __name__ == '__main__':
    mean_shift()
