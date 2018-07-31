#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/20
# @Author  : Wenhao Shan
# @Dsc     : Convolutional Neural Network with TensorFlow

import numpy as np
import tensorflow as tf

BATCH_SIZE: int = 128


class CNN:
    """
    卷积神经网络训练
    """

    def __init__(self, input_data_tr_x, input_data_tr_y, input_data_va_x, input_data_va_y,
                 input_data_te_x, input_data_te_y):
        self.weight = None  # 第一个卷积层的权重
        self.b_offset = None    # 第一个卷积层的偏置
        self.weight_2 = None    # 第二个卷积层的权重
        self.b_offset_2 = None  # 第二个卷积层的偏置
        self.weight_3 = None    # 第三个卷积层的权重
        self.b_offset_3 = None  # 第三个卷积层的偏置
        self.weight_4 = None    # 全连接层中输入层到隐含层的权重
        self.b_offset_4 = None  # 全连接层中输入层到隐含层的偏置
        self.weight_o = None    # 隐含层到输出层的权重
        self.b_offset_o = None  # 隐含层到输出层的偏置
        self.p_keep_convolution = None     # 卷积层中样本保持不变的比例
        self.p_keep_hidden = None   # 全连接层中样本保持不变的比例
        self.tr_x = input_data_tr_x     # 训练数据中的特征
        self.tr_y = input_data_tr_y     # 训练数据中的标签
        self.va_x = input_data_va_x     # 验证数据中的特征
        self.va_y = input_data_va_y     # 验证数据中的标签
        self.te_x = input_data_te_x     # 测试数据中的特征
        self.te_y = input_data_te_y     # 测试数据中的标签

    def fit(self):
        x = tf.placeholder("float", [None, 28, 28, 1])
        y = tf.placeholder("float", [None, 10])

        # 第一层卷积核大小为3*3, 输入一张图, 输出32个feature map.
        # tf.random_normal: 从正太分布中随机初始权重, 标准差为0.01
        self.weight = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        self.b_offset = tf.Variable(tf.constant(0.0, shape=[32]))   # tf.constant 创建常量
        print("-----------------", self.weight)
        # 第二层卷积核大小为3*3, 输入32个feature map, 输出64个feature map
        self.weight_2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        self.b_offset_2 = tf.Variable(tf.constant(0.0, shape=[64]))
        # 第三层卷积核大小为3*3, 输入64个feature map, 输出128个feature map
        self.weight_3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        self.b_offset_3 = tf.Variable(tf.constant(0.0, shape=[128]))
        # FC 128 * 4 * 4 inputs, 625 outputs
        self.weight_4 = tf.Variable(tf.random_normal([128 * 4 * 4, 625], stddev=0.01))
        self.b_offset_4 = tf.Variable(tf.constant(0.0, shape=[625]))
        # FC 625 inputs, 10 outputs(labels)
        self.weight_o = tf.Variable(tf.random_normal([625, 10], stddev=0.01))
        self.b_offset_o = tf.Variable(tf.constant(0.0, shape=[10]))

        self.p_keep_convolution = tf.placeholder("float")   # 卷积层的dropout概率
        self.p_keep_hidden = tf.placeholder("float")    # 全连接层的dropout概率

        # 第一个卷积层: padding=SAME, 保证输出的feature map 与输入矩阵的大小相同
        # tf.nn.relu: 使用relu激活函数
        # tf.nn.conv2d: 卷积, 详情(https://www.cnblogs.com/lovephysics/p/7220111.html)
        # padding参数选择算法: (https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t)
        # conv2d实际做了以下操作:
        # 1.将filter, 即self.weight转为二维矩阵, shape为[filter_height * filter_width * in_channels, output_channels]
        # 2.从input tensor(x)中提取image patches, 每个patch是一个virtual tensor, shape[batch, out_height, out_width,
        # filter_height * filter_width * in_channels]
        # 3.将每个filter矩阵和image patch向量相乘.
        l_c_1 = tf.nn.relu(tf.nn.conv2d(x, self.weight, strides=[1, 1, 1, 1], padding="SAME")
                           + self.b_offset)  # shape(?, 28, 28, 32)
        # max_pooling, 窗口大小为2*2
        l_p_1 = tf.nn.max_pool(l_c_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")   # shape(?, 14, 14, 32)
        # dropout: 每个神经元有p_keep_convolution的概率以1/p_keep_convolution的比例进行归一化,
        # 否则置为0
        l_1 = tf.nn.dropout(l_p_1, self.p_keep_convolution)

        # 第二个卷积层
        l_c_2 = tf.nn.relu(tf.nn.conv2d(l_1, self.weight_2, strides=[1, 1, 1, 1], padding="SAME")
                           + self.b_offset_2)   # shape(?, 14, 14, 64)
        l_p_2 = tf.nn.max_pool(l_c_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")    # shape(?, 7, 7, 64)
        l_2 = tf.nn.dropout(l_p_2, self.p_keep_convolution)

        # 第三个卷积层
        l_c_3 = tf.nn.relu(tf.nn.conv2d(l_2, self.weight_3, strides=[1, 1, 1, 1], padding="SAME")
                           + self.b_offset_3)  # shape(?, 7, 7, 128)
        l_p_3 = tf.nn.max_pool(l_c_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")    # shape(?, 4, 4, 128)

        # 将所有的feature map合并成一个2048维向量
        l_3 = tf.reshape(l_p_3, [-1, self.weight_4.get_shape().as_list()[0]])    # shape(?, 2018)
        l_3 = tf.nn.dropout(l_3, self.p_keep_convolution)

        # 后面两层为全连接层, tf.matmul为矩阵相乘
        l_4 = tf.nn.relu(tf.matmul(l_3, self.weight_4) + self.b_offset_4)
        l_4 = tf.nn.dropout(l_4, self.p_keep_hidden)

        pyx = tf.matmul(l_4, self.weight_o) + self.b_offset_o

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pyx, labels=y))  # 交叉熵目标函数
        # 使用RMSProp算法最小化目标函数
        train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
        predict_op = tf.argmax(pyx, 1)  # 返回每个样本的预测结果

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for i in range(30):
                # zip合并两个列表, 返回zip对象, 存放的类似于[[0, 128], [128, 256],...]的数据, 使得训练集分步对模型进行
                # 训练和优化
                training_batch = zip(range(0, len(self.tr_x), BATCH_SIZE),
                                     range(BATCH_SIZE, len(self.tr_x) + 1, BATCH_SIZE))
                for start, end in training_batch:
                    # feed_dict占位符为计算提供数据
                    sess.run(train_op, feed_dict={x: self.tr_x[start: end], y: self.tr_y[start: end],
                                                  self.p_keep_convolution: 0.8, self.p_keep_hidden: 0.5})
                if i % 3 == 0:
                    corr = np.mean(np.argmax(self.va_y, axis=1) ==
                                   sess.run(predict_op, feed_dict={x: self.va_x, y: self.va_y, self.p_keep_convolution: 1.0, self.p_keep_hidden: 1.0}))
                    print("Accuracy at step %s on validation set: %s " % (i, corr))

            # 最终在测试集上的输出
            corr_te = np.mean(np.argmax(self.te_y, axis=1) ==
                              sess.run(predict_op, feed_dict={x: self.te_x, y: self.te_y, self.p_keep_convolution: 1.0, self.p_keep_hidden: 1.0}))
            print("Accuracy on test set : %s " % corr_te)


def cnn_run():
    """
    CNN run
    :return:
    """
    from tensorflow.examples.tutorials.mnist import input_data
    # 1、导入数据集
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)   # 读取数据
    # mnist.train.images是一个55000 * 784(784为一张图的所有像素点的像素值矩阵记录)维的矩阵, mnist.train.labels是一个55000 * 10 维的矩阵
    tr_x, tr_y, va_x, va_y, te_x, te_y = mnist.train.images, mnist.train.labels, mnist.validation.images, \
        mnist.validation.labels, mnist.test.images, mnist.test.labels
    tr_x = tr_x.reshape(-1, 28, 28, 1)  # 将每张图片用一个28 * 28(将784转换为28 * 28矩阵像素存储)的矩阵表示, (55000, 28, 28, 1)
    va_x = va_x.reshape(-1, 28, 28, 1)
    te_x = te_x.reshape(-1, 28, 28, 1)
    # 2、训练CNN模型
    cnn = CNN(tr_x, tr_y, va_x, va_y, te_x, te_y)
    cnn.fit()


if __name__ == '__main__':
    cnn_run()
