#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/21
# @Author  : Wenhao Shan
# @Dsc     :  some base function and tool of machine learning, for example the Sigmoid function.

import time
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


class PaintingWithLabel(BasePainting):
    """
    画图类(带有标签的数据)
    """
    def __init__(self, fig_support=111, name=""):
        super(PaintingWithLabel, self).__init__(fig_support, name)

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


class PaintingNoLabel(BasePainting):
    """
    画图(用于回归)
    """
    def __init__(self, fig_support=111, name=""):
        super(PaintingNoLabel, self).__init__(fig_support, name)

    def painting_simple(self, feature: list, label: list, index: int):
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

    def painting_with_offset(self, data: list, label: list):
        """
        带偏置的画图
        :param data: 原始数据
        :param label:
        :return:
        """
        self.painting_simple(data, label, 1)

    def painting_no_offset(self, data: np.mat, label: list):
        """
        不带偏置的画图
        :param data: 原始数据
        :param label:
        :return:
        """
        self.painting_simple(data, label, 0)