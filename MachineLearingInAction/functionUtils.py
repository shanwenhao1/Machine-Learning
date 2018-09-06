#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/6
# @Author  : Wenhao Shan
# @Dsc     : Some base function and tool of ML in action

import matplotlib.pyplot as plt
from numpy import zeros, ndarray, tile
from mpl_toolkits.mplot3d import Axes3D


# ------------------------------------------ Normalized Data ------------------------------------------ #

class HM:
    """
    归一化方法
    """
    @classmethod
    def average(cls, data_set: ndarray):
        """
        线性函数归一化
        :param data_set:
        :return: return_set(ndarray): 均一化后特征值
        """
        min_val = data_set.min(0)       # 获取矩阵每列最小元素, 格式为: 1 * n
        max_val = data_set.max(0)
        ranges = max_val - min_val      # 最大值与最小值之间的差值
        m = data_set.shape[0]           # m * n...矩阵
        # date_set矩阵中每第1-n列减去该列最小元素(线性归一化), tile为copy矩阵函数,
        # tile(np.array([0, 1]), (2, 2))得到np.array([0, 1, 0, 1,], [0, 1, 0, 1,])
        return_set = data_set - tile(min_val, (m, 1))
        return_set = return_set / tile(ranges, (m, 1))
        return return_set, ranges, min_val


# ------------------------------------------ Painting ------------------------------------------ #

class BasePainting:
    """
    Base class of Painting
    """
    def __init__(self, fig_support: int=111, name: str=""):
        # linux系统下需要加上matplotlib.use('Agg')
        # AGG is the abbreviation of Anti-grain geometry engine.
        # matplotlib.use('Agg')
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


class Painting3D(BasePainting):
    """
    paint slot of ndarray
    """
    def __init__(self, fig_support: int=111, name: str=""):
        super(Painting3D, self).__init__(fig_support, name)
        self.ax1 = Axes3D(self.fig)

    def paint(self, data_mat: ndarray, data_labels: list):
        all_type_dict = dict()
        one_type_dict = {"x_data": list(), "y_data": list(), "z_data": list()}
        for _, label in enumerate(data_labels):
            if label not in all_type_dict.keys():
                all_type_dict[label] = one_type_dict
        for i in range(data_mat.shape[0]):
            type_dict = all_type_dict[data_labels[i]]
            type_dict["x_data"].append(data_mat[i][0])
            type_dict["y_data"].append(data_mat[i][1])
            type_dict["z_data"].append(data_mat[i][2])
            all_type_dict[data_labels[i]] = type_dict
        i = 0
        for key, value in all_type_dict.items():
            self.ax1.scatter(value["x_data"], value["y_data"], value["z_data"],
                             c=self.all_color[i], marker=self.all_marker[i])
            i += 1


# ------------------------------------------ Load Data ------------------------------------------ #


class BaseLoadData:
    """
    Base Class of Load Data
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def __enter__(self):
        self.f = open(self.file_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()


class LD(BaseLoadData):
    """
    Load Data To numpy.ndarray
    """
    def __init__(self, file_path: str):
        super(LD, self).__init__(file_path)

    def load_to_ndarray(self, feature_len: int, need_label: bool=True):
        """
        导入数据, 格式为numpy.ndarray格式
        :param feature_len: 特征个数
        :param need_label: 是否有标签(默认为每行最后一个数据)
        :return:
        """
        index: int = 0
        if need_label:
            label_list: list = list()

        # 得到文件行数
        array_lines = self.f.readlines()
        data_length = len(array_lines)
        return_mat = zeros((data_length, feature_len))    # data_length * feature_len矩阵初始化, 以0填充
        for line in array_lines:
            line = line.strip()     # 去空格
            line_split_list = line.split('\t')
            return_mat[index, :] = line_split_list[0: feature_len]
            if need_label:
                label_list.append(int(line_split_list[-1]))
            index += 1
        return return_mat, label_list if need_label else return_mat

