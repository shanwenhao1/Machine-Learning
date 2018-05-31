#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/18
# @Author  : Wenhao Shan
# @Des: the base use of Keras in Deep Learning, just like hello world in programming

import keras

# step1: define a set of function

# keras 函数用法链接 https://blog.csdn.net/u012969412/article/details/70882296
model = keras.Sequential()
# input_dim: 指定输入数据的维度; unit: 大于0的整数, 代表全连接嵌入的维度
model.add(keras.layers.Dense(500, input_dim=28 * 28))
model.add(keras.layers.Activation("sigmoid"))
# more deep classify
model.add(keras.layers.Dense(500))
model.add(keras.layers.Activation("sigmoid"))
# output
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation("softmax"))

# step2: goodness of function

# step3.1: configuration
# loss: 参考损失函数; optimizer: 参考优化器; metrics: 包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=['accuracy']
# SGD随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量。
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.1), metrics=["accuracy"])

# step3.2: find the optimal network parameters
# x: 输入数据,numpy array或者list类型, list内为numpy array; y: 标签, numpy array类型;
# batch seize: 指定进行梯度下降时每个batch包含的样本数, 训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步;
# epochs: 训练的轮数
model.fit(x="training data(images)", y="Labels", batch_size=100, epochs=20)


# 验证模式的准确性, x: 测试集(同model.fit参数), y: 测试数据的标签集(同model.fit参数)
score = model.evaluate(x="testing data set", y="testing label set")
print("Total loss on Testing Set: ", score[0])
print("Accuracy of Testing Set: ", score[1])

# 利用模型进行预测
result = model.predict(x="one testing data")