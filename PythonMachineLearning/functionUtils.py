#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/21
# @Author  : Wenhao Shan
# @Dsc     :  some base function and tool of machine learning, for example the Sigmoid function.

import numpy as np
import matplotlib.pyplot as plt
from utils.errors import ActionError


def sig(x):
    """
    Sigmoid 函数
    :param x: x(mat): feature * w(权重)
    :return: sigmoid(x) (mat): Sigmoid值
    """
    return 1.0 / (1 + np.exp(-x))


def partial_sig(x: np.mat or float):
    """
    Sigmoid导函数的值
    :param x: x: 自变量, 可以是矩阵或者是任意实数
    :return: out(mat or float): Sigmoid导函数的值
    """
    m, n = np.shape(x)
    out = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            # σ'(x) = σ(x)(1 - σ(x)) 推导公式见P125或者自己可以简单推导一下
            out[i, j] = sig(x[i, j]) * (1 - sig(x[i, j]))
    return out


def least_square(feature: np.mat, label: np.mat):
    """
    最小二乘法
    :param feature: 特征
    :param label: 标签
    :return:
    """
    w = (feature.T * feature).I * feature.T * label
    return w


def get_list_from_mat(mat_data: np.mat, offset: int=0, need_mul: bool=False):
    """
    将mat数据转换为list数据
    :param mat_data:
    :param offset: 偏置
    :param need_mul: 是否需要嵌套list
    :return:
    """
    if not need_mul:
        list_data = [mat_data[position, offset] for position in range(np.shape(mat_data)[0])]
    else:
        list_data = [[mat_data[position, offset]] for position in range(np.shape(mat_data)[0])]
    return list_data


class BasePainting:
    """
    画图基类
    """

    def __init__(self, fig_support: int=111, name: str=""):
        self.fig = plt.figure(1)
        self.ax1 = self.fig.add_subplot(fig_support)
        self.ax1.set_title(name)
        self.all_color = ["black", "brown", "gold", "blue", "red", "maroon", "yellow", "gray"]
        self.all_marker = ["x", "o", "+", "*", "h", "s", "^", "D"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 必须要用plt.ion()否则plt.close()关闭不了窗口, 详情请见 http://bbs.eetop.cn/thread-382878-1-1.html
        plt.ion()
        plt.show()
        # 暂停3秒钟
        plt.pause(3)
        plt.close()


class PaintingWithMat(BasePainting):
    """
    画图类(带有标签的数据)
    """

    def __init__(self, fig_support=111, name=""):
        super(PaintingWithMat, self).__init__(fig_support, name)

    def painting(self, input_point: np.mat, label: np.mat, index_1: int, index_2: int):
        """
        针对(x, y, label)型数据画图, index_1和index_2参数根据是否有偏置项进行调节
        :param input_point:
        :param label:
        :param index_1:
        :param index_2:
        :return:
        """
        # 所有数据类型
        all_point_type = [label[index, 0] for index in range(np.shape(label)[0])]
        # set去除重复的元素
        all_point_type = set(all_point_type)
        for position, _type in enumerate(all_point_type):
            x_scatter = [input_point[index, index_1] for index in range(np.shape(input_point)[0]) if label[index, 0] == _type]
            y_scatter = [input_point[index, index_2] for index in range(np.shape(input_point)[0]) if label[index, 0] == _type]
            if position >= len(self.all_color):
                raise ActionError("颜色不够请提升配置")
            self.ax1.scatter(np.array(x_scatter), np.array(y_scatter), c=self.all_color[position],
                             marker=self.all_marker[position])

    def painting_with_offset(self, input_point: np.mat, label: np.mat):
        """
        针对np.mat型数据(x, y, label)
        :param input_point: (offset, x, y)
        :param label: (label)
        :return:
        """
        self.painting(input_point, label, 1, 2)

    def painting_with_no_offset(self, input_point: np.mat, label: np.mat):
        """
        无偏置项的打印
        :param input_point: (x, y)
        :param label: (label)
        :return:
        """
        self.painting(input_point, label, 0, 1)

    def painting_simple_index(self, data: list):
        """
        画图(data中的数据为[x, y, label]型的数据)
        :param data:
        :return:
        """
        self.painting_simple(data, self.ax1)

    def painting_after_train(self, original_data: list, after_data: list):
        """
        训练后分类的画图
        :param original_data: 原数据
        :param after_data: 之后的数据
        :return:
        """
        ax2 = self.fig.add_subplot(212)
        self.painting_simple(original_data, self.ax1)

        after_original_data = original_data.copy()
        for i in range(len(after_data)):
            after_original_data[i][-1] = after_data[i]
        self.painting_simple(after_original_data, ax2)

    def painting_simple(self, data, ax):
        """
        简单画图调用
        :param data:
        :param ax:
        :return:
        """
        # 所有数据类型
        all_point_type = [index[-1] for index in data]
        # set去重
        all_point_type = set(all_point_type)
        for position, _type in enumerate(all_point_type):
            x_scatter = list()
            y_scatter = list()
            for index in data:
                if index[-1] == _type:
                    x_scatter.append(index[0])
                    y_scatter.append(index[1])
            if position >= len(self.all_color):
                raise ActionError("颜色配置不够")
            ax.scatter(np.array(x_scatter), np.array(y_scatter), c=self.all_color[position],
                       marker=self.all_marker[position])


class PaintingWithList(BasePainting):
    """
    画图(用于回归)
    """

    def __init__(self, fig_support=111, name=""):
        super(PaintingWithList, self).__init__(fig_support, name)

    def painting_mul_list(self, feature: list, label: list, index: int):
        """
        画图(单纯的描点), 用于回归
        :param feature:
        :param label:
        :param index: offset位移
        :return:
        """
        x_scatter = np.array([_data[index] for _data in feature])
        y_scatter = np.array(label)
        self.ax1.scatter(x_scatter, y_scatter, c=self.all_color[1], marker=self.all_marker[1])

    def painting_simple_list(self, feature: list, label: list):
        """
        画图(feature和label)都是一维list
        :param feature:
        :param label:
        :return:
        """
        x_scatter = np.array(feature)
        y_scatter = np.array(label)
        self.ax1.scatter(x_scatter, y_scatter, c=self.all_color[1], marker=self.all_marker[1])

    def painting_with_offset(self, data: list, label: list, mul_simple: bool=False):
        """
        带偏置的画图
        :param data: 原始数据
        :param label:
        :param mul_simple: feature是否是一维数组型的list
        :return:
        """
        if not mul_simple:
            self.painting_mul_list(data, label, 1)
        else:
            self.painting_simple_list(data, label)

    def painting_no_offset(self, data: list, label: list, mul_simple: bool=False):
        """
        不带偏置的画图
        :param data: 原始数据
        :param label:
        :param mul_simple:
        :return:
        """
        if not mul_simple:
            self.painting_mul_list(data, label, 0)
        else:
            self.painting_simple_list(data, label)


class LoadData:
    """
    导入数据的类
    """

    def __init__(self, file_name: str, feature_type: str="float", label_type: str="float"):
        self.file_name = file_name
        self.feature_type = feature_type
        self.label_type = label_type
        self.all_type_turn = {"int": int, "float": float}
        if self.feature_type not in self.all_type_turn.keys():
            raise ActionError("feature Type Error")
        if self.label_type not in self.all_type_turn.keys():
            raise ActionError("Label Type Error")

    def load_data(self, offset: int=None, need_label_length: bool=False, need_list: bool=False, feature_end: int=1):
        """
        导入数据(训练或测试数据)
        :param offset: 偏置项
        :param need_label_length: 是否需要标签的个数
        :param need_list: 是否需要list型数据
        :param feature_end: feature获取的截止位置
        :return: feature(mat or list) 特征
                  label(mat or list) 标签
        """
        f = open(self.file_name)
        feature_data = list()
        label_array_list = list()   # 标签需要转换为二维数组的list
        label_list = list()     # 标签只需要一个list
        for line in f.readlines():
            feature_tmp = list()
            label_tmp = list()
            lines = line.strip().split("\t")
            if offset:
                feature_tmp.append(offset)  # 偏置项
            for i in range(len(lines) - feature_end):
                feature_tmp.append(self.all_type_turn[self.feature_type](lines[i]))
            label_tmp.append(self.all_type_turn[self.label_type](lines[-1]))
            label_list.append(self.all_type_turn[self.label_type](lines[-1]))

            feature_data.append(feature_tmp)
            label_array_list.append(label_tmp)
        f.close()
        if need_label_length:
            if need_list:
                return feature_data, label_list, len(set(label_list))
            else:
                # set消除重复元素
                return np.mat(feature_data), np.mat(label_list), len(set(label_list))
        return np.mat(feature_data), np.mat(label_array_list)

    def load_data_with_limit(self, number: int, offset: int=None):
        """
        导入测试数据
        :param number:
        :param offset: 偏置项
        :return:
        """
        f = open(self.file_name)
        feature_data = list()
        for line in f.readlines():
            feature_tmp = list()
            lines = line.strip().split("\t")
            if len(lines) != number - 1:
                continue
            if offset:
                feature_tmp.append(offset)
            for x in lines:
                feature_tmp.append(self.all_type_turn[self.feature_type](x))
            feature_data.append(feature_tmp)
        f.close()
        return np.mat(feature_data)


class SaveModel:
    """
    保存模型类
    """

    def __init__(self, file_name: str):
        self.file_name = file_name

    def save_model(self, w: np.mat):
        """
        保存模型, 一维
        :param w: 权重值
        :return:
        """
        m, n = np.shape(w)
        w_array = [str(w[i, 0]) for i in range(m)]
        self.f.write("\t".join(w_array))

    def save_model_mul(self, w: np.mat):
        """
        保存模型, 多维
        :param w:
        :return:
        """
        m, n = np.shape(w)
        for i in range(m):
            w_tmp = [str(w[i, j]) for j in range(n)]
            self.f.write("\t".join(w_tmp) + "\n")

    def __enter__(self):
        self.f = open(self.file_name, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()


class LoadModel:
    """
    加载模型
    """

    def __init__(self, file_name: str):
        self.file_name = file_name

    def load_model(self, need_transpose=False):
        """
        导入模型
        :param need_transpose: 是否需要转置
        :return: weight(mat): 权重值
        """
        w = [float(line.strip()) for line in self.f.readlines()]
        weight = np.mat(w)
        return weight if not need_transpose else weight.T

    def load_model_mul(self, need_transpose=False):
        """
        导入模型(需要多维weight)
        :param need_transpose: 是否需要转置
        :return: weight(mat): 权重值
        """
        w = list()
        for line in self.f.readlines():
            lines = line.strip().split("\t")
            w_tmp = [float(x) for x in lines]
            w.append(w_tmp)
        weight = np.mat(w)
        return weight if not need_transpose else weight.T

    def __enter__(self):
        self.f = open(self.file_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()
