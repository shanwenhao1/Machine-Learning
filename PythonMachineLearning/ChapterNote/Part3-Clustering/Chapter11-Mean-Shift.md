# Mean Shift

MeanShift算法又被称为均值漂移算法, 用来处理类别个数未知的数据集. 基于距离的聚类算法.

## Mean Shift向量
![](../MularGif/Part3-Clustering/Chapter11Gif/Mean%20Shift%20Vector.gif), 其中S<sub>h</sub>指的是一个半径为h的高维球区域
S<sub>h</sub>的定义为: ![](../MularGif/Part3-Clustering/Chapter11Gif/Mean%20Shift%20Roud.gif), 计算漂移均值向量通过
计算圆S<sub>h</sub>中的每一个样本点X<sup>(i)</sup>相对于点X的偏移量(X<sup>(i)</sup>-X), 再对所有的漂移均值向量求和
再求平均.
- 由于每一个样本点X<sup>(i)</sup>对于样本X的贡献不一样, 因此引入核函数度量贡献

### 核函数

高斯核函数![](../MularGif/Part3-Clustering/Chapter11Gif/Gaussian%20Kernel%20Function.gif)(类似于正太分布函数), 当
带宽h一定时, 样本点之间的距离越大, 其核函数值就越小. (相当于基于贡献度) 

### MeanShift算法原理

向基本的MeanShift向量中增加核函数, 得到改进后的Mean Shift向量形式: 
<br><center>![](../MularGif/Part3-Clustering/Chapter11Gif/MSV%20With%20Gaussian.gif)</center></br>
,其中K((X<sup>(i)</sup> - X)/h)是核函数

MeanShift通过迭代的方式, 对每一个样本点计算其漂移值, 以计算出来的漂移均值点作为新的起始点, 直到得到最终的均值漂移点即为
最终的聚类中心. 实际上利用了概率密度, 求得概率密度的局部最优解
    - 公式推导P212, 可知Mean Shift向量与概率密度函数的梯度成正比, 因此Mean Shift向量总是指向概率密度增加的方向.

算法步骤:
- 在指定的区域内计算每一个样本点的漂移均值
- 移动该点到漂移均值点处
- 重复上述的过程(计算新的漂移均值, 移动)
- 当满足最终的条件时, 则退出

Mean Shift向量的修正
- 公式: ![](../MularGif/Part3-Clustering/Chapter11Gif/Correction%20Mean%20Shift.gif)
- 修正后的算法流程
    - 计算m<sub>h</sub>(X)
    - 令X=m<sub>h</sub>(X)
    - 如果||m_h(X) - X|| < ε, 结束循环, 否则, 重复上述步骤.