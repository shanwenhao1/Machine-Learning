#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/11
# @Author  : Wenhao Shan
# @Dsc     : base learning of MNIST by TensorFlow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 60000行的训练数据集(mnist.train)和10000行的测试数据集(mnist.test)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
x = tf.placeholder("float", [None, 784])                                # 每一张图展开成784维的向量
# 维度10是用来表示0-9的分类, 比如0:[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)                                  # w * x + b

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))                          # 交叉熵loss function

# 由于TensorFlow拥有一张描述你各个计算单元的图, 因此它可以自动地使用反向传播算法来有效确定你的变量
# 是如何影响你想要最小化的那个成本值的, 然后TensorFlow会用你选择的优化算法来不断地修改变量以降低成本

# 这里我们使用梯度下降算法, 以0.01的学习速率最小化交叉熵

# TensorFlow实际所做的是, 在后台给描述你计算的那张图内增加一系列新的计算操作单元用于实现反向传播算法
# 和梯度下降算法. 然后, 它返回给你的只是一个单一的操作, 当运行这个操作时, 它用梯度下降算法训练你的模型,
# 微调你的变量, 不断减少成本
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)                    # 随机抓取训练数据中的100个批处理数据点
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# 评估模型
# tf.argmax给出某个tensor对象在某一维上的其数据最大值所在的索引值, 在这里是筛选出结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))        # tf.equal检测预测是否匹配正确
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))         # tf.cast将bool转换成float, 并用tf.reduce_mean求平均值
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
