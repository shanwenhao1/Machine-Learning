# 支持向量机(Support Vector Machine)

## 前言
- 与分割超平面距离最近的样本称为支持向量(Support Vector)
- 在确定最终的分隔超平面时, 只有支持向量起作用, 其他的样本点并不起作用.

核心思想: 将低维数据线性不可分映射成高纬度数据成为线性可分

## 感知机算法

直接使用通误分类的样本到分隔超平面之间的距离S作为损失函数, 并利用梯度下降法求得
误分类的损失函数的极小值, 得到分隔超平面.

样本到分隔超平面的距离为: 
<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/Perceptron%20algorithm.gif)</center></br> 
其中||W||为W的L<sub>2</sub>范式, [L<sub>2</sub>范式详解](https://blog.csdn.net/zchang81/article/details/70208061)

感知机算法的损失函数为: ![](../MularGif/Part1-Classification/Chapter4Gif/Perceptron%20Loss.gif)

感知机算法存在的问题: 采用不同的初始值或者不同的分类点, 最终产生的分隔超平面是不同的.

## SVM算法

SVM解出的分隔超平面不仅能够正确划分训练数据集, 而且几何间隔最大.

训练样本中, 与分隔超平面距离最近的样本称为支持向量(Support Vector), 在确定最终的分隔超平面中, 只有支持向量
起作用, 因此这类模型被称为支持向量机.

### 函数间隔与几何间隔

#### 函数间隔(Functional Margin)

当预测值W·X<sup>(i)</sup>+b和样本标签y<sup>(i)</sup>同号时, 表明最终的分类是正确的.

定义分隔超平面有关样本点的函数间隔为: ![](../MularGif/Part1-Classification/Chapter4Gif/Function%20Distance.gif), 
因此定义分隔超平面为求所有样本点至分隔超平面的函数间隔最小值

函数间隔可以表示分类预测的正确性和准确性

#### 几何间隔(Geometrical Margin)

函数间隔中当W和b变化时, 函数间隔也跟着变化. 因此引入几何间隔, 实际上是在函数间隔的基础上加上了均一化操作.
<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/Geometric%20interval.gif)</center></br>

### 推导步骤

支持向量的约束条件(其实就是几何距离大于或等于支持向量到超平面的几何距离): 
<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/Restrictions.gif)</center></br>
引入松弛向量ξ<sub>i</sub>的目的是为了解决部分线性不可分的样本不满足大于或等于1的约束条件

#### 引入拉格朗日乘数法解决最优问题求解, 将原始的带约束的优化问题转换成其对偶问题.
- 1/||W||的最大值问题相当于1/2 ||W||<sup>2</sup>
- 对偶问题往往更容易求解
- 可以自然的引入核函数, 进而推广到非线性分类问题

公式: 
<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/Lagrangian%20Dual.gif)</center></br>
, 向量α和β为拉格朗日乘子向量.(对于非支持向量α都是等于0的, 因为非支持向量对超平面无影响)

拉格朗日对偶问题, 最小优化问题即变为: 
<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/Lagrangian%20Dual%20problem.gif)</center></br>
利用对偶性, 对换max和min的位置(由于α为不等式约束, 带有α的去求解w,b,ξ过于难).先固定α求解
![](../MularGif/Part1-Classification/Chapter4Gif/min.gif), 分别求导得:
<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/result.gif)</center></br>
- 将上述结果代入至![](../MularGif/Part1-Classification/Chapter4Gif/min.gif), 得到:
<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/min%20L.gif)</center></br>
- 再对![](../MularGif/Part1-Classification/Chapter4Gif/min.gif)求α的极大, 即可求得最优的α,就可求出权重W
<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/max.gif)</center></br>
- 求解b的最优解时, 公式: ![](../MularGif/Part1-Classification/Chapter4Gif/b.gif)

#### 由线性支持向量机到非线性支持向量机(引入核函数)

引入高斯核函数解决非线性支持向量机问题
- 核函数通过将数据映射到高维空间, 来解决在原始空间中线性不可分的问题
<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/GaussianKernelFunction.gif)</center>,
ψ是从X到内积特征空间F的映射</br>

之所以不先映射再根据内积的公式计算,而是采用核函数的原因是:
- 前一种在高维中计算, 计算复杂度指数级增长
- 后一种直接在原来的低维空间计算, 而不需要显示的写出映射后的结果

序列最小最优化算法SMO(Sequential Minimal Optimization)求解: 书69页
- 思想: 将一个大的问题划分成一系列小的问题, 通过对子问题的求解, 达到对对偶问题的求解过程


#### SMO算法(Sequential Minimal Optimization)

序列最小最优化算法SMO的思想将一个大的问题划分成一系列小的问题, 通过对子问题的求解, 达到对对偶问题的求解过程.
- SMO不断将对偶问题的二次规划问题分解为只有两个变量的二次规划子问题. 因此属于二次规划优化算法
    - SMO算法只优化两个变量, 将其他变量都视为常数, 由于
    ![](../MularGif/Part1-Classification/Chapter4Gif/Restrictions%20alpha.gif), 如果固定α<sub>3</sub>,α<sub>4</sub>,
    ...,α<sub>m</sub>, 那么α<sub>1</sub>,α<sub>2</sub>之间的关系也确定了.
    - 因此优化函数则变为了以下形式:
    <br><center>![](../MularGif/Part1-Classification/Chapter4Gif/SMO.gif)</center></br>
    - 最后得解α
    <br><center>![](../MularGif/Part1-Classification/Chapter4Gif/New%20alpha.gif)</center></br>
- 第二个变量的选择
    - SMO算法称选择第二一个变量为内层循环,假设我们在外层循环已经找到了α1α1, 第二个变量α2α2的选择标准
    是让|E1−E2|有足够大的变化.由于α1α1定了的时候,E1E1也确定了,所以要想|E1−E2|最大,
    只需要在E1E1为正时,选择最小的EiEi作为E2E2, 在E1E1为负时,选择最大的EiEi作为E2E2,可以将所有的EiEi保存下来加快迭代.
    - 如果内存循环找到的点不能让目标函数有足够的下降, 可以采用遍历支持向量点来做α2α2,直到目标函数有足够的下降,
     如果所有的支持向量做α2α2都不能让目标函数有足够的下降,可以跳出循环,重新选择α1α1　
- 计算阈值b和差值E<sub>i</sub>:
<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/b%20and%20E.png)</center></br>

##### SMO算法总结

<br><center>![](../MularGif/Part1-Classification/Chapter4Gif/SMO%20summary.png)</center></br>
     
## 杂谈

- [SVM算法详解](https://blog.csdn.net/v_july_v/article/details/7624837)
- [支持向量机学习笔记](https://blog.csdn.net/v_victor/article/details/51508884)
- [SMO优化求解](https://www.cnblogs.com/pinard/p/6111471.html)
- [KKT与拉格朗日乘子法](https://blog.csdn.net/xianlingmao/article/details/7919597)
    - KKT条件是拉格朗日乘子法的泛化
    - 拉格朗日乘子法是求解条件极值的解析方法, 可将所有约束的优化模型问题转化为无约束极值问题的求解.
