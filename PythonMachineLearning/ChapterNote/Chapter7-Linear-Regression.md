# 线性回归(Linear Regression)

## 前言

线性回归算法是很多算法的基础, 但对于样本之间存在很高的相关性, 基本线性回归算法很难得到泛化能力较高的模型

线性回归算法是一种全局的回归算法, 对局部的拟合效果并不好

## 线性回归

线性回归方程: ![](MularGif/Chapter7Gif/LinearRegressionFormular.gif), 其中b为偏置, w<sub>i</sub>为回归系数

损失函数: 有绝对损失函数和平方损失函数, 由于平方损失处处可导, 因此通常使用平方误差作为线性回归模型的损失函数.
公式为: 
<br><center>![](MularGif/Chapter7Gif/LRLossFunction.gif)</center>, 求解希望得到平方误差的最小值</br>


### 最小二乘法

又称最小平方法是一种数学优化技术, 通过最小化误差的平方和寻找数据的最佳函数匹配.利用最小二乘法可以简便地求得未知的数据, 
并使得这些求得的数据与实际数据之间误差的平方和为最小

## 杂谈

- [最小二乘法](https://blog.csdn.net/quicmous/article/details/51705125): 采用最下二乘法的原因是可以求导且方程易解
