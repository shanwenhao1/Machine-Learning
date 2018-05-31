#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/24
# @Author  : Wenhao Shan

# 运算符模块
from numpy import *

from utils.errors import ActionError


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
    def _classify0(inx, dataset: ndarray, labels: list, k: int):
        """
        KNN分类, 具体步骤:
            第一步: 先将待判定数据用tile函数复制成样本数据相同的格式, 如样本数据为4 * 2的二维数组 , 则待定数据也copy成
                    4*2的二维数组(其中每个1*2数组都为待判定数据)
            第二步: 求出每个样本数据跟待判定数据的得距离, 并排序
            第三步: 按照k值选择出跟待判定数据最接近的k个样本点, 根据这k个样本点所属类型判断待判定点的类型(样本所属类型
            最多的类型)
        :param inx: 1 * 2 的二维数组
        :param dataset: 数据集
        :param labels: 标签
        :param k: k值选择(用于选择最近邻居的数目)
        :return:
        """
        # -------------------------------------------距离计算
        # .shape求二维数组的size, 结果为(4, 2)
        data_set_size = dataset.shape[0]
        # tile(a: array, b: int/tuple)重复a b(int)次 或(2, 3)行2次列三次
        diff_mat = tile(inx, (data_set_size, 1)) - dataset
        # 表示对数组中的每个数做平方, 如[[1, -0.1]]求平方后变为[[1, 0.01]]
        sq_diff_mat = diff_mat ** 2
        # 将矩阵的每一行向量相加, 如sum([[0,1,2],[2,1,3],axis=1) 结果为: array([3, 6])
        sq_distances = sq_diff_mat.sum(axis=1)
        # 对平方和进行开根号(此步骤可以考虑省略, 因为平方和可以直接作比较)
        distances = sq_distances ** 0.5
        # .argsort()返回排序后的索引值, 其中靠前的表示越靠近样本的点(升序排序)
        sorted_dist_indicie = distances.argsort()

        # 取k个靠近样本点附近的点的类别, 进行计数
        class_count = dict()
        for i in range(k):
            vote_ilabel = labels[sorted_dist_indicie[i]]
            class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
        # 根据计数来判定样本点所属类别
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    @classmethod
    def file_matrix(cls, filename):
        """
        读取文件数据集, 保存为numpy.ndarray格式数据
        :param filename:
        :return:
        """
        file = open(filename)
        # 得到文件行数
        array_lines = file.readlines()
        data_length = len(array_lines)
        # 创建以0填充的 data_length * 3 的NumPy矩阵
        return_mat = zeros((data_length, 3))
        class_label_vector = list()
        index = 0
        for line in array_lines:
            line = line.strip()
            list_form_line = line.split('\t')
            # [index, :] 表示取得矩阵第index行所有元素
            return_mat[index, :] = list_form_line[0: 3]
            class_label_vector.append(int(list_form_line[-1]))
            index += 1
        return return_mat, class_label_vector

    def depart_movie(self, inx: list, k: int):
        """
        判断电影类型
        :param inx:
        :param k:
        :return:
        """
        result = self._classify0(inx, self.group, self.labels, k)
        return result

    def depart_appointment(self, file_name: str, ax_type: int, ay_type: int):
        """
        判断约会对象是否合适
        :return:
        """
        # linux系统下需要加上matplotlib.use('Agg')
        # AGG is the abbreviation of Anti-grain geometry engine.
        # matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dating_data_mat, dating_labels = self.file_matrix(file_name)
        # 散点图使用dating_data_mat矩阵的第二(玩游戏所耗时间百分比)、第三列数据(每周消费冰淇淋公升数)
        # 15.0 * array(dating_labels)为设置颜色(利用了dating_labels中的标签来画不同的颜色)
        # 其中self.auto_norm()归一化数据
        ax.scatter(dating_data_mat[:, ax_type], dating_data_mat[:, ay_type],
                   15.0 * array(dating_labels), 15.0 * array(dating_labels))
        plt.show()

    @staticmethod
    def auto_norm(dataset: ndarray):
        """
        归一化数据, 以使不同类型的数据的权重不会因为数值不同而受到影响
        :param dataset:
        :return:
        """
        min_val = dataset.min(0)
        max_val = dataset.max(0)
        ranges = max_val - min_val
        norm_data_set = zeros(shape(dataset))
        m = dataset.shape[0]
        norm_data_set = dataset - tile(min_val, (m, 1))
        # numpy中矩阵除法需要使用函数linalg.solve(matA, matB), 因此下方是矩阵内的数相除, 而不是矩阵相除
        norm_data_set = norm_data_set / tile(ranges, (m, 1))
        return norm_data_set, ranges, min_val


def classifyPerson(file_name: str, k_number: int):
    """
    约会网站预测函数, 根据用户输入的对方信息和以往该用户对对象看法进行分类
    :return:
    """
    result_list = ["not at all", "in small doses", "in large doses"]
    percent_tats = float(input("percentage of time spent playing video games?"))
    fly_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = KnnLearning.file_matrix(file_name)
    norm_mat, ranges, min_val = KnnLearning.auto_norm(dating_data_mat)
    in_arr = array([fly_miles, percent_tats, ice_cream])
    classifier_result = KnnLearning._classify0((in_arr - min_val) / ranges, norm_mat, dating_labels, k_number)
    print("You will probably like this person: ", result_list[classifier_result - 1])


if __name__ == '__main__':
    # # 模拟数据集
    # group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # # 模拟标签
    # labels = ['A', 'A', 'B', 'B']
    # KnnLearning(group, labels).run([-1, 1], 3)
    os_path = os.getcwd()
    file_path = os_path + "/datingTestSet2.txt"
    # KnnLearning().depart_appointment(file_path, Appointment.PlayGame, Appointment.FlyMile)
    classifyPerson(file_path, 3)
