# 支持向量机(Support Vector Machine)

## 前言
- 与分割超平面距离最近的样本称为支持向量(Support Vector)
- 在确定最终的分隔超平面时, 只有支持向量起作用, 其他的样本点并不起作用.

核心思想: 将低维数据线性不可分映射成高纬度数据成为线性可分

## SVM算法

### 推导步骤
约束条件: 
<br><center>![](MularGif/Chapter4Gif/Restrictions.gif)</center></br>

引入拉格朗日乘数法解决最优问题求解, 将原始的带约束的优化问题转换成其对偶问题.
<br><center>![](MularGif/Chapter4Gif/GaussianKernelFunction.gif)</center>,ψ是从X到内积特征空间F的映射</br>

引入高斯核函数解决非线性支持向量机问题
- 核函数通过将数据映射到高维空间, 来解决在原始空间中线性不可分的问题

序列最小最优化算法SMO(Sequential Minimal Optimization)求解: 书69页
- 思想: 将一个大的问题划分成一系列小的问题, 通过对子问题的求解, 达到对对偶问题的求解过程

## 杂谈
- [分类算法详解](https://www.cnblogs.com/berkeleysong/articles/3251245.html)
- [支持向量机学习笔记](https://blog.csdn.net/v_victor/article/details/51508884)
