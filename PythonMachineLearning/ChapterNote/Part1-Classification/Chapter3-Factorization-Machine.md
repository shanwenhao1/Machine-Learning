# Factorization Machine

## 前言

FM是基于LR算法进行扩展(针对线性不可分问题), 基于矩阵分解的算法

可处理以下三类问题:
- 回归问题(Regression): 损失函数可取为 最小平方误差
- 二分类问题(Binary Classification): 损失函数可取为logit loss函数
- 排序(Ranking)

## 主要优点
- 用于高度稀疏数据场景
- 具有线性的计算复杂度

## FM机器学习
[公式推导及引入交叉项的原因](https://blog.csdn.net/itplus/article/details/40534923)

- 损失函数(二分类因子分解机FM算法)
[logit loss公式](https://blog.csdn.net/google19890102/article/details/79496256):
<br><center>![](../MularGif/Part1-Classification/Chapter3Gif/LossMular.gif)</center></br>
其中σ(x) 为sigmoid函数, 由公式可知 ![](../MularGif/Part1-Classification/Chapter3Gif/comparey.gif)损失函数就越小

- 引入交叉项:
<br><center>![](../MularGif/Part1-Classification/Chapter3Gif/CrossCoeifficient.gif)</center></br>
交叉项系数w<sub>i,j</sub>不能直接加在交叉项x<sub>i</sub>x<sub>j</sub>前(由于稀疏数据的原因), 需要引入矩阵分解的
思想引入辅助向量V<sub>i</sub>

- 模型的求解:
<br><center>![](../MularGif/Part1-Classification/Chapter3Gif/ModuleSolution.gif)</center></br>


## 随机梯度下降(Stochastic Gradient Descent)
SGD算法步骤![SGD算法步骤](../MularGif/Part1-Classification/Chapter3Gif/SGD.png)

SGD针对数据量特别大的情况, 由于梯度下降算法使用所有的样本进行模型参数的学习, 需花费大量的计算成本.
因此, 提出SGD简化计算, 提高效率

特点: 在每次迭代过程中, 仅根据一个样本对模型中的参数进行调整. 每次迭代只是考虑让该样本点的J(θ)
趋向最小, 而不管其他的样本点. 这样算法会很快, 但是收敛过程会比较曲折, 整体上只能接近局部最优解, 
而无法真正达到局部最优解.

对loss function公式求导: ![](../MularGif/Part1-Classification/Chapter3Gif/LossMularDerivative%20.gif)

## 杂谈

除了利用FM算法外, 对于大部分的线性不可分问题, 可以人工对特征进行处理(如核函数处理)转换为线性可分问题, 
再用LR算法进行二分类。

理论分析中, 我们要求参数k取得足够大, 但是在高度稀疏数据场景中, 由于没有足够的样本来估计复杂的交互矩阵, 
因此k通常应取得很小.事实上, 对于参数k(亦即FM的表达能力)的限制, 在一定程度上可以提高模型的泛化能力.

- [FM及FFM](https://blog.csdn.net/asd136912/article/details/78318563)
- [FM详解](https://blog.csdn.net/liruihongbob/article/details/75008666)
- [比较好的算法原理详解(含采样数据处理)](https://blog.csdn.net/itplus/article/details/40536025)
