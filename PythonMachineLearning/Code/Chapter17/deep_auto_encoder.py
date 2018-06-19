#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/19
# @Author  : Wenhao Shan
# @Dsc     : Use De-Noising AutoEncoder to build Deep neural network

import numpy as np
import tensorflow as tf
from PythonMachineLearning.Code.Chapter17.auto_encoder import DeNoisingAutoEncoder


class StackedDeNosingAutoEncoder:
    def __init__(self, hidden_list, input_data_train_x, input_data_train_y, input_data_valid_x, input_data_valid_y,
                 input_data_test_x, input_data_test_y, corruption_level=0.3):
        self.encode_w = list()  # 保存网络中每一层的权重
        self.encode_b = list()  # 保存网络中每一层的偏置
        self.hidden_list = hidden_list  # 每一个隐含层的节点个数
        self.input_data_train_x = input_data_train_x    # 训练样本的特征
        self.input_data_train_y = input_data_train_y    # 训练样本的标签
        self.input_data_valid_x = input_data_valid_x    # 验证样本的特征
        self.input_data_valid_y = input_data_valid_y    # 验证样本的标签
        self.input_data_test_x = input_data_test_x      # 测试样本的特征
        self.input_data_test_y = input_data_test_y      # 测试样本的标签

    def fit(self):
        """
        堆叠降噪自编码器类的训练
        :return:
        """
        # 1、训练每一个降噪自编码器
        next_input_data = self.input_data_train_x
        for i, hidden_size in enumerate(self.hidden_list):
            print("-------------------- train the %s sda --------------------" % (i + 1))
            dae = DeNoisingAutoEncoder(hidden_size, next_input_data)
            dae.fit()
            w_eval, b_eval, encode_eval = dae.get_value()
            self.encode_w.append(w_eval)
            self.encode_b.append(b_eval)
            next_input_data = encode_eval

        # 2、堆叠多个降噪自编码器
        n_input = self.input_data_train_x.shape[1]
        n_output = self.input_data_train_y.shape[1]

        x = tf.placeholder("float", [None, n_input], name="X")
        y = tf.placeholder("float", [None, n_output], name="Y")

        encoding_w_tmp = list()
        encoding_b_tmp = list()

        last_layer = None
        layer_nodes = list()
        encoder = x
        for i, hidden_size in enumerate(self.hidden_list):
            # 以每一个自编码器的值作为初始值
            encoding_w_tmp.append(tf.Variable(self.encode_w[i], name='enc-w-{}'.format(i)))
            encoding_b_tmp.append(tf.Variable(self.encode_b[i], name="enc-b-{}".format(i)))
            encoder = tf.nn.sigmoid(tf.matmul(encoder, encoding_w_tmp[i]) + encoding_b_tmp[i])  # 公式y = sigmoid(wx + b)
            layer_nodes.append(encoder)
            last_layer = layer_nodes[i]

        # 加入少量的噪声来打破对称性以及避免0梯度, tf.truncated_normal从截断的正态分布中输出随机值.
        # tensor.get_shape()返回一个元组, TensorShape([Dimension(2), Dimension(3)])
        # (https://blog.csdn.net/fireflychh/article/details/73692183)
        last_w = tf.Variable(tf.truncated_normal([last_layer.get_shape()[1].value, n_output], stddev=0.1),
                             name="sm-weights")
        last_b = tf.Variable(tf.constant(0.1, shape=[n_output]), name="sm-biases")
        last_out = tf.matmul(last_layer, last_w) + last_b
        layer_nodes.append(last_out)

        # tf.reduce_mean求均值, tf.nn.softmax_cross_entropy_with_logits: 对最后一层的输出做softmax, 再将结果与样本实际值
        # 做一个交叉熵即为loss 详情: https://blog.csdn.net/mao_xiao_feng/article/details/53382790
        cost_sme = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=last_out, labels=y))   # loss function
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost_sme)  # 梯度下降的方法求解损失函数

        model_predictions = tf.argmax(last_out, 1)
        correct_predictions = tf.equal(model_predictions, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))    # tf.cast将tensor改变成新的类型

        # 3、微调
        tr_x = self.input_data_train_x
        tr_y = self.input_data_train_y
        va_x = self.input_data_valid_x
        va_y = self.input_data_valid_y
        te_x = self.input_data_test_x
        te_y = self.input_data_test_y
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for i in range(50):
                for start, end in zip(range(0, len(tr_x), 128), range(128, len(tr_x) + 1, 128)):
                    sess.run(train_step, feed_dict={x: tr_x[start: end], y: tr_y[start: end]})  # 利用梯度下降训练模型
                if i % 5 == 0:
                    print("Accuracy at step %s on validation set: %s " % (i, sess.run(accuracy,
                                                                                      feed_dict={x: va_x, y: va_y})))
                    print("Accuracy on test set: %s" % (sess.run(accuracy, feed_dict={x: te_x, y: te_y})))


def stacked_de_noising_auto_encoder():
    """
    堆叠自编码器run
    :return:
    """
    from tensorflow.examples.tutorials.mnist import input_data
    # 1、导入数据集(手写体数字识别数据集), 详情: https://www.cnblogs.com/eczhou/p/7860508.html
    # 自动将MNIST数据集划分为train, validation和test三个数据集
    min_st = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # 2、训练SDAE模型, min_st获取的数据中每个数据共有10个标签, 其中label标签为1表示该图片为数字几,
    # image像素矩阵中不为0的地方表示像素的大小, 取值范围为[0,1]
    sda = StackedDeNosingAutoEncoder([1000, 1000, 1000], min_st.train.images, min_st.train.labels,
                                     min_st.validation.images, min_st.validation.labels, min_st.test.images,
                                     min_st.test.labels)
    sda.fit()


if __name__ == '__main__':
    stacked_de_noising_auto_encoder()
