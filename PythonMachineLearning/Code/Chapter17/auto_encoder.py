#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/18
# @Author  : Wenhao Shan
# @Dsc     : De-Noising AutoEncoder

import numpy as np
import tensorflow as tf


class DeNoisingAutoEncoder:
    """
    降噪编码器基类
    """

    def __init__(self, n_hidden: int, input_data, corruption_level=0.3):
        self.weight = None  # 输入层到隐含层的权重
        self.b_offset = None    # 输入层到隐含层的偏置
        self.encode_r = None   # 隐含层的输出
        self.layer_size = n_hidden  # 隐含层的节点个数
        self.input_data = input_data    # 输入样本
        self.keep_prob = 1 - corruption_level   # 特征保持不变的比例
        self.w_eval = None  # 权重W的值
        self.b_eval = None  # 偏置b的值

    def fit(self):
        """
        降噪自编码器的训练
        :return:
        """
        n_visible = self.input_data.shape[1]    # 输入层节点的个数
        # 输入一张图片用28 * 28 = 784的向量表示, tf.placeholder变量初始化时使用占位符
        x = tf.placeholder("float", [None, n_visible], name="X")
        # 用于将部分输入数据置为0
        mask = tf.placeholder("float", [None, n_visible], name="mask")
        # 创建权重和偏置
        w_init_max = 4 * np.sqrt(6. / (n_visible + self.layer_size))    # 权重初始化选择区间
        # 从均匀分布的区间内初始化权重
        w_init = tf.random_uniform(shape=[n_visible, self.layer_size], minval=-w_init_max, maxval=w_init_max)

        # 编码器, 变量初始化并赋值
        self.weight = tf.Variable(w_init, name="W")     # 784 * 500
        self.b_offset = tf.Variable(tf.zeros([self.layer_size]), name="b")  # 隐含层的偏置
        # 解码器
        w_prime = tf.transpose(self.weight)     # tf.transpose转置函数
        b_prime = tf.Variable(tf.zeros([n_visible]), name="b_prime")

        tilde_x = mask * x  # 对输入样本加入噪声

        # 输出, 使用公式y = sigmoid(w * x + b)
        y = tf.nn.sigmoid(tf.matmul(tilde_x, self.weight) + self.b_offset)  # 隐含层的输出(编码过程), tf.matmul矩阵相乘
        z = tf.nn.sigmoid(tf.matmul(y, w_prime) + b_prime)  # 重构输出(解码过程)

        cost = tf.reduce_mean(tf.pow(x - z, 2))     # 均方误差(定义的损失函数), 可以考虑加入正则提高鲁棒性
        # 最小化均方误差, 根据损失函数使用梯度下降的方法优化, 0.01为学习率即步长α
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        tr_x = self.input_data
        # 开始训练
        with tf.Session() as sess:
            # 初始化所有的参数
            tf.initialize_all_variables().run()
            for i in range(30):
                # 按照步长为128截取数据集分步计算.
                for start, end in zip(range(0, len(tr_x), 128), range(128, len(tr_x) + 1, 128)):
                    input_ = tr_x[start: end]   # 设置输入
                    # 设置mask, binomial二项分布随机采样, 按照概率将隐含层的输入置为0, 得到含噪音的数据, 提高鲁棒性.
                    # (模拟实际中的图像部分像素被遮挡、文本因记录原因漏掉了一些单词)
                    mask_np = np.random.binomial(1, self.keep_prob, input_.shape)
                    # 开始训练(利用梯度下降)
                    sess.run(train_op, feed_dict={x: input_, mask: mask_np})
                if i % 5.0 == 0:
                    mask_np = np.random.binomial(1, 1, tr_x.shape)
                    print("Loss Function at step %s is %s" % (i, sess.run(cost, feed_dict={x: tr_x, mask: mask_np})))
            # 保存好输入层到隐含层的参数
            self.w_eval = self.weight.eval()
            self.b_eval = self.b_offset.eval()
            mask_np = np.random.binomial(1, 1, tr_x.shape)
            self.encode_r = y.eval({x: tr_x, mask: mask_np})    # 隐含层输出

    def get_value(self):
        """
        获取降噪自编码器的参数
        :return: self.w_eval(mat): 输入层到隐含层的权重
                  self.b_eval(mat): 输入层到隐含层的偏置
                  self.encode_r(mat): 隐含层的输出值
        """
        return self.w_eval, self.b_eval, self.encode_r
