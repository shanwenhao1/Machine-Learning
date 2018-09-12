#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/12
# @Author  : Wenhao Shan
# Dsc      : Plot Tree
import matplotlib.pyplot as plt


class PaintingTree:
    """
    The class is used to painting the tree that you input
    """

    def __init__(self):
        # 定义文本框和箭头格式
        self.decision_node = dict(boxstyle="sawtooth", fc="0.8")
        self.leaf_node = dict(boxstyle="round4", fc="0.8")
        self.arrow_args = dict(arrowstyle="<-")
        self.total_width = 0.0                        # 树的宽度(用于将树绘制在中心位置)
        self.total_height = 0.0                       # 树的深度
        # self.plot_tree.x0ff和self.plot_tree.y0ff用来追踪已经绘制的节点位置, 以及放置下一个节点的位置
        # 实际上我们是通过叶子节点的数目将x轴划分为若干个部分
        self.tree_x0ff = 0
        self.tree_y0ff = 0
        self.fig = plt.figure(1, facecolor='white')
        self.fig.clf()
        ax_props = dict(xticks=[], yticks=[])
        self.plt = plt.subplot(111, frameon=False, **ax_props)

    def create_plot(self, in_tree: dict):
        """
        绘制树图入口
        :param in_tree:
        :return:
        """
        # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo
        # puropses
        self.total_width = float(self.get_num_leafs(in_tree))
        self.total_height = float(self.get_tree_depth(in_tree))
        self.tree_x0ff = -0.5 / self.total_width
        self.tree_y0ff = 1.0
        self.plot_tree(in_tree, (0.5, 1.0), '')
        plt.show()

    def plot_tree(self, my_tree: dict, parent_pt: tuple, node_text: str):
        """

        :param my_tree:
        :param parent_pt:
        :param node_text:
        :return:
        """
        # 获取树的宽度和深度
        num_leafs = self.get_num_leafs(my_tree)
        depth = self.get_tree_depth(my_tree)

        first_str = list(my_tree.keys())[0]
        cnt_pt = (self.tree_x0ff + (1.0 + float(num_leafs)) /
                  2.0 / self.total_width, self.tree_y0ff)
        self.plot_mid_text(cnt_pt, parent_pt, node_text)
        # 标记子节点属性值
        self.plot_node(first_str, cnt_pt, parent_pt, self.decision_node)
        #
        second_dict = my_tree[first_str]
        # 按比例减少全局变量self.tree_y0ff, 并标志此处要绘制子节点
        self.tree_y0ff = self.tree_y0ff - 1.0 / self.total_height
        for key in second_dict.keys():
            if type(
                    second_dict[key]).__name__ == 'dict':                   # 递归画子节点的图
                self.plot_tree(second_dict[key], cnt_pt, str(key))
            else:                                                           # 如果是叶子节点则画图
                self.tree_x0ff = self.tree_x0ff + 1.0 / self.total_width
                self.plot_node(
                    second_dict[key],
                    (self.tree_x0ff,
                     self.tree_y0ff),
                    cnt_pt,
                    self.leaf_node)
                self.plot_mid_text(
                    (self.tree_x0ff, self.tree_y0ff), cnt_pt, str(key))
        self.tree_y0ff = self.tree_y0ff + 1.0 / self.total_height

    def get_num_leafs(self, my_tree: dict):
        """
        计算树的宽度(最底层叶子节点的总数量)
        :param my_tree:
        :return:
        """
        num_leafs = 0
        # 当前节点名称(父节点)
        first_str = list(my_tree.keys())[0]
        second_dict = my_tree[first_str]                            # 当前节点的叶子节点
        for key in second_dict.keys():
            if type(
                    second_dict[key]).__name__ == 'dict':           # 如果还存在叶子节点, 则递归下去
                num_leafs += self.get_num_leafs(second_dict[key])
            else:
                num_leafs += 1
        return num_leafs

    def get_tree_depth(self, my_tree):
        """
        计算树的深度(多少层)
        :param my_tree:
        :return:
        """
        max_depth = 0
        # 当前节点名称(父节点)
        first_str = list(my_tree.keys())[0]
        second_dict = my_tree[first_str]                            # 当前节点的叶子节点
        for key in second_dict.keys():
            if type(
                    second_dict[key]).__name__ == 'dict':           # 如果还存在叶子节点, 则递归下去
                this_depth = 1 + self.get_tree_depth(second_dict[key])
            else:
                this_depth = 1
            if this_depth > max_depth:
                max_depth = this_depth
        return max_depth

    def plot_node(
            self,
            node_txt: str,
            center_pt: tuple,
            parent_pt: tuple,
            node_type: dict):
        """
        绘制带箭头的注解
        :param node_txt:
        :param center_pt:
        :param parent_pt:
        :param node_type:
        :return:
        """
        self.plt.annotate(
            node_txt,
            xy=parent_pt,
            xycoords='axes fraction',
            xytext=center_pt,
            textcoords='axes fraction',
            va="center",
            ha="center",
            bbox=node_type,
            arrowprops=self.arrow_args)

    def plot_mid_text(self, cnt_rpt: tuple, parent_pt: tuple, txt_str: str):
        """
        计算当前父节点和子节点的中间位置
        :param cnt_rpt:
        :param parent_pt:
        :param txt_str:
        :return:
        """
        x_mid = (parent_pt[0] - cnt_rpt[0]) / 2.0 + cnt_rpt[0]
        y_mid = (parent_pt[1] - cnt_rpt[1]) / 2.0 + cnt_rpt[1]
        # 添加标签信息
        self.plt.text(
            x_mid,
            y_mid,
            txt_str,
            va="center",
            ha="center",
            rotation=30)


if __name__ == '__main__':
    from MachineLearingInAction.Code.Chapter3_Random_Forest.trees import create_tree, create_data_set
    my_dat, labels = create_data_set()
    my_tree = create_tree(my_dat, labels)
    PaintingTree().create_plot(my_tree)
