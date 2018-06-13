#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/13
# @Author  : Wenhao Shan
# @Dsc     : Label Propagation, One Algorithm Of Community partitioning

import numpy as np


def load_data(file_path: str):
    """
    导入数据
    :param file_path: 文件位置
    :return: vector_dict(dict): 节点: 社区
              edge_dict(dict): 存储节点之间的边和权重
    """
    f = open(file_path)
    vector_dict = dict()    # 存储节点
    edge_dict = dict()  # 存储边
    for line in f.readlines():
        lines = line.strip().split("\t")

        for i in range(2):
            if lines[i] not in vector_dict.keys():  # 节点已存储
                # 将节点放入到vector_dict中, 设置所属社区为其自身
                vector_dict[lines[i]] = int(lines[i])
                # 将边放入到edge_dict
                edge_list = list()
            else:  # 节点未存储
                edge_list = edge_dict[lines[i]]

            if len(lines) == 3:
                edge_list.append(lines[1 - i] + ":" + lines[2])
            else:
                edge_list.append(lines[1 - i] + ":" + "1")
            edge_dict[lines[i]] = edge_list
    return vector_dict, edge_dict


def get_max_community_label(vector_dict: dict, adjacency_node_list: list):
    """
    得到相邻的节点中标签数最多的标签
    :param vector_dict: 节点: 社区
    :param adjacency_node_list: 节点的邻接节点
    :return: 节点所属的社区
    """
    label_dict = dict()
    for node in adjacency_node_list:
        node_id_weight = node.strip().split(":")
        node_id = node_id_weight[0]  # 邻接节点id
        node_weight = int(node_id_weight[1])    # 与邻接节点之间的权重
        if vector_dict[node_id] not in label_dict.keys():
            label_dict[vector_dict[node_id]] = node_weight
        else:
            label_dict[vector_dict[node_id]] += node_weight

    # 找到最大的标签并返回
    sort_list = sorted(label_dict.items(), key=lambda d: d[1], reverse=True)
    return sort_list[0][0]


def check(vector_dict: dict, edge_dict):
    """
    检查是否满足终止条件
    :param vector_dict: 节点: 社区
    :param edge_dict: 存储节点之间的边和权重
    :return: 是否需要更新
    """
    for node in vector_dict.keys():
        adjacency_node_list = edge_dict[node]   # 与节点node相连接的节点
        node_label = vector_dict[node]  # 节点node所属社区
        label = get_max_community_label(vector_dict, adjacency_node_list)
        if node_label == label:     # 遍历所有节点, 确定其所属的社区标签是最大的, 则满足终止条件, 否则继续
            continue
        else:
            return False
    return True


def label_propagation(vector_dict: dict, edge_dict: dict):
    """
    标签传播, 异步更新方式
    :param vector_dict: 节点: 社区
    :param edge_dict: 存储节点之间的边和权重
    :return: vector_dict(dict): 节点: 社区
    """
    # 初始化, 设置每个节点属于u不同的社区
    t = 0
    # 以随机的次序处理每个节点
    while True:
        if check(vector_dict, edge_dict):   # 满足终止条件, 退出
            break
        t += 1
        print("iteration %d" % t)
        # 对每一个node进行更新
        for node in vector_dict.keys():
            adjacency_node_list = edge_dict[node]   # 获取节点node的邻接节点
            vector_dict[node] = get_max_community_label(vector_dict, adjacency_node_list)
        print(vector_dict)
    return vector_dict


def lb_run():
    """
    Label Propagation
    :return:
    """
    # 1、导入数据
    print("----------1.load data ------------")
    vector_dict, edge_dict = load_data("cd_data.txt")
    print("original community: \n", vector_dict)
    # 2、利用label propagation算法进行社区划分
    print("----------2.label propagation ------------")
    vec_new = label_propagation(vector_dict, edge_dict)
    # 3、最终的社区划分的结果
    print("final_result:", vec_new)


if __name__ == '__main__':
    lb_run()
