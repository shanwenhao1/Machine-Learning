#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/24
# @Author  : Wenhao Shan

from numpy import *

# 随机生成一个4 * 4的数组
random_array = random.rand(4, 4)

# 调用mat函数将数组转化为矩阵
rand_mat = mat(random_array)

# .I实现矩阵求逆, eye(4)能够创建4 * 4的单位矩阵
inverse_matrix_rand_mat = rand_mat.I

# 矩阵相乘
power_matrix = rand_mat * inverse_matrix_rand_mat
