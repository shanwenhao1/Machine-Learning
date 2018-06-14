# 机器学习

## 绪论
- 监督学习
    - 具体流程
        - 训练部分: 获取数据 → 特征提取 → 监督学习 → 评价 → 监督学习 → 评价(多个循环后) → 预测部分
        - 预测部分: 模型 → 预测
- 监督学习
- 推荐算法
    - 随着信息量的急剧扩大, 信息过载问题变得尤为突出. 推荐信息给用户帮助用户从海量的数据中找到感兴趣、有价值的信息.
    - 主要算法有: 协同过滤算法、基于矩阵分解的推荐算法、基于图的推荐算法.
- 深度学习
    - 概念: 逐层训练
    - 常见模型
        - 自编码器模型: 通过堆叠自编码器构建深层网络
        - 卷积神经网络模型: 通过卷积层与采样层的不断交替构建深层网络
        - 循环神经网络

## 章节笔记

### Part Ⅰ 分类算法(Classification algorithm)

标签是一些离散的值, 代表着不同的分类

- [第一章](ChapterNote/Part1-Classification/Chapter1-Logistic-Regression.md) Logistic Regression
- [第二章](ChapterNote/Part1-Classification/Chapter2-Softmax-Regression.md) Softmax Regression
- [第三章](ChapterNote/Part1-Classification/Chapter3-Factorization-Machine.md) Factorization Machine
- [第四章](ChapterNote/Part1-Classification/Chapter4-Support-Vector-Machine.md) Support Vector Machine
- [第五章](ChapterNote/Part1-Classification/Chapter5-Random-Forest.md) Random Forest
- [第六章](ChapterNote/Part1-Classification/Chapter6-Back-Propagation.md) Back Propagation Neural Network

### Part Ⅱ 回归算法(Regression algorithm)
标签是一些连续的值, 目标是通过训练得到样本特征到连续标签的映射

- [第七章](ChapterNote/Part2-Regression/Chapter7-Linear-Regression.md) Linear Regression
- [第八章](ChapterNote/Part2-Regression/Chapter8-RidgeAndLasso-Regression.md) Ridge Regression and Lasso Regression
- [第九章](ChapterNote/Part2-Regression/Chapter9-CART-Regression.md) Classification And Regression Tree

### Part Ⅲ 聚类算法(Clustering Algorithm)

聚类算法是一种典型的无监督学习算法, 通过定义不同的相似性的度量方法, 将具有相似属性的事物聚集到同一个类中.
只包含样本的特征, 不包含样本的标签信息.
- [第十章](ChapterNote/Part3-Clustering/Chapter10-K-Means.md) K-Means
- [第十一章](ChapterNote/Part3-Clustering/Chapter11-Mean-Shift.md) Mean Shift
- [第十二章](ChapterNote/Part3-Clustering/Chapter12-DBSCAN.md) DBSCAN
- [第十三章](ChapterNote/Part3-Clustering/Chapter13-Label-Propagation.md) Label Propagation

### Part Ⅳ 推荐算法(Recommended algorithm)

在现如今的大数据时代, 爆炸式增长的数据, 给用户带来了信息过载的痛苦. 推荐算法的出现给用户提供了很多的
便利, 并且精准的推荐对于企业来说也是更高效的选择.
- [第十四章](ChapterNote/Part4-Recommendation/Chapter14-Collaborative-Filtering.md) Collaborative Filtering

### Part Ⅴ 深度学习(Deep Learning)

### Part Ⅵ 项目实践(Machine Learning In Action)

## 杂谈

- [回归和线性](https://blog.csdn.net/hzw19920329/article/details/77200475)
- [重点: Sparsity and Some Basics of L1 Regularization](http://freemind.pluskid.org/machine-learning/sparsity-and-some-basics-of-l1-regularization/#ed61992b37932e208ae114be75e42a3e6dc34cb3) 