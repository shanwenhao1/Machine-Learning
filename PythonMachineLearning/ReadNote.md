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

- [第一章](ChapterNote/Chapter1-Logistic-Regression.md) Logistic Regression
- [第二章](ChapterNote/Chapter2-Softmax-Regression.md) Softmax Regression
- [第三章](ChapterNote/Chapter3-Factorization-Machine.md) Factorization Machine
- [第四章](ChapterNote/Chapter4-Support-Vector-Machine.md) Support Vector Machine
- [第五章](ChapterNote/Chapter5-Random-Forest.md) Random Forest
- [第六章](ChapterNote/Chapter6-Back-Propagation.md) Back Propagation Neural Network

### Part Ⅱ 回归算法(Regression algorithm)
标签是一些连续的值, 目标是通过训练得到样本特征到连续标签的映射

- [第七章](ChapterNote/Chapter7-Linear-Regression.md) Linear Regression
- [第八章](ChapterNote/Chapter8-RidgeAndLasso-Regression.md) Ridge Regression and Lasso Regression

### Part Ⅲ 聚类算法(Clustering Algorithm)

### Part Ⅳ 推荐算法(Recommended algorithm)

### Part Ⅴ 深度学习(Deep Learning)

### Part Ⅵ 项目实践(Machine Learning In Action)

## 杂谈

- [回归和线性](https://blog.csdn.net/hzw19920329/article/details/77200475)
- [重点: Sparsity and Some Basics of L1 Regularization](http://freemind.pluskid.org/machine-learning/sparsity-and-some-basics-of-l1-regularization/#ed61992b37932e208ae114be75e42a3e6dc34cb3) 