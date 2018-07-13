# Mean Shift

MeanShift算法又被称为均值漂移算法, 用来处理类别个数未知的数据集. 基于距离的聚类算法. 聚类中心是通过在给定区域中的样本的
均值来确定的, 通过不断更新聚类中心, 直到最终的聚类中心不再改变为止. Mean Shift在聚类、图像平滑、分割和视频跟踪等方面
有广泛的应用.

## Mean Shift向量

![](../MularGif/Part3-Clustering/Chapter11Gif/Mean%20Shift%20Vector.gif),
X为球中心点(计算漂移均值向量实际上是在寻找更优聚类中心点). 其中S<sub>h</sub>指的是一个半径为h的高维球区域
S<sub>h</sub>的定义为: ![](../MularGif/Part3-Clustering/Chapter11Gif/Mean%20Shift%20Roud.gif).</br>
在计算漂移均值向量的过程中, 通过计算圆S<sub>h</sub>中的每一个样本点X<sup>(i)</sup>相对于点X的偏移量
(X<sup>(i)</sup>-X), 再对所有的漂移均值向量求和再求平均.
- 由于每一个样本点X<sup>(i)</sup>对于样本X的贡献不一样, 因此引入核函数度量贡献

### 核函数

高斯核函数![](../MularGif/Part3-Clustering/Chapter11Gif/Gaussian%20Kernel%20Function.gif)(类似于正太分布函数), 当
带宽h一定时, 样本点之间的距离越大, 其核函数值就越小. (相当于基于贡献度) 

### MeanShift算法原理

向基本的MeanShift向量中增加核函数, 得到改进后的Mean Shift向量形式: 
<br><center>![](../MularGif/Part3-Clustering/Chapter11Gif/MSV%20With%20Gaussian.gif)</center></br>
,其中K((X<sup>(i)</sup> - X)/h)是核函数

MeanShift通过迭代的方式, 对每一个样本点计算其漂移值, 以计算出来的漂移均值点作为新的起始点, 直到得到最终的均值漂移点即为
最终的聚类中心.

#### Mean Shift算法的解释

Mean Shift实际上利用了概率密度, 求得概率密度的局部最优解

#### 概率密度梯度
对一个概率密度梯度函数f(X), 已知d维空间中n个采样点X<sup>i</sup>, i=1,...,n, f(X)的核函数估计为:
<br><center>![](../MularGif/Part3-Clustering/Chapter11Gif/Probability%20density%20kernel%20function.gif)</center></br>
其中, K(x)是一个单位核函数, K(x)可表示为k(||x||<sup>2</sup>), 因此概率密度函数的梯度可由求导得知:
<br><center>![](../MularGif/Part3-Clustering/Chapter11Gif/Kernel%20Function%20Derivatives.gif)</center></br>
其中第一个方括号内是以G(X)为核函数对概率密度函数f(X)的估计, 第二个括号中的是Mean Shift向量, 则f(X)的梯度可以表示为:</br>
<br><center>![](../MularGif/Part3-Clustering/Chapter11Gif/Mean%20Shift.gif)</center></br>
由上式可知, Mean Shift向量M<sub>h</sub>(X)与概率密度函数f(X)的梯度成正比, 因此Mean Shift向量总是指向概率密度增加的方向.

Mean Shift向量的修正
- 公式: ![](../MularGif/Part3-Clustering/Chapter11Gif/Correction%20Mean%20Shift.gif)
- 修正后的算法流程
    - 计算m<sub>h</sub>(X)
    - 令X=m<sub>h</sub>(X)
    - 如果||m_h(X) - X|| < ε, 结束循环, 否则, 重复上述步骤.

算法步骤:
- 在指定的区域内计算每一个样本点的漂移均值
- 移动该点到漂移均值点处
- 重复上述的过程(计算新的漂移均值, 移动)
- 当满足最终的条件时, 则退出


## 杂谈

- [原文链接](https://blog.csdn.net/google19890102/article/details/51030884)