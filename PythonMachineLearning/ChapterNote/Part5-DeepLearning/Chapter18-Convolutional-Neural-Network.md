# 卷积神经网络

## 前言

卷积神经网络(Convolutional Neural Network, CNN)是多层神经网络模型的一个变种, 在图像领域得到了广泛的应用.
充分利用图像数据在局部上的相关性, 减少了网络中参数的个数, 方便求解

传统神经网络模型中需要训练的参数过于庞大, 而CNN则避免了这种问题

## 卷积神经网络

### 核心概念

- 稀疏连接(Sparse Connectivity)
    - 主要是通过对数据中的局部区域进行建模, 以发现局部的一些特性
    - 减少了网络中边的数量
- 共享权值(Shared Weights)
    - 为了简化计算的过程, 使得需要优化的参数变少, 加快学习速度
    - 在CNN网络中, 每一组感知野的权值共享
- 池化(Pooling)
    - 子采样的目的是解决图像中的平移不变性, 即所要识别的内容与其在图像中的具体位置无关.
    - 一般采用最大池化(max-pooling): 将输入图像划分成一系列不重叠的正方形区域, 然后对于每一个子
    区域, 输出其中的最大值.(能够进一步降低计算量, 通过消耗非最大值, 为上层降低了计算量).
    
CNN通过在邻接层的神经元之间使用局部连接来发现输入特征在空间上存在的局部相关性.

感受野: 第m层节点的输入是第m-1层的神经元的一个子集, 被选择的子集的大小被称为感受野.

卷积神经网络中, 利用卷积层和下采样层的交替叠加, 得到特征的高级抽象, 再对高级抽象的特征进行全连接的映射, 
最终对其进行分类.


### 卷积神经网络求解

重要的三层:
- 卷积层(Convolution Layer): 卷积操作
- 下采样层(Sub-Sampling Layer):  max-pooling操作
- 全连接层(Fully-Connected Layer): 全连接的MLP操作

#### 卷基层

卷积操作主要是f(x)g(x)在重合区域的积分, 求解卷积操作中的权重的值采用
- 信号的正向传播: 输入与权重的卷积操作, 比如2 * 2的卷积核: 
![](../MularGif/Part5-DeepLearning/Chapter18Gif/Positive%20Propagation.gif)
- 误差的反向传播: 权重的梯度是输入与误差矩阵的卷积操作

一维数据下的卷积定义:![](../MularGif/Part5-DeepLearning/Chapter18Gif/Convolution%20One.gif)

二维数据下的卷积中, 权重的组成是矩阵, 不是向量, 定义如下: 
<br><center>![](../MularGif/Part5-DeepLearning/Chapter18Gif/Convolution%20Two.gif)</center></br>
通过原始数据与权重矩阵的点积, 得到卷积后的结果

三维数据下的卷积操作, 是二维数据下的卷积操作的推广形式. 比如一张RGB图片, 对应了3个通道, 对每一个通道采用二维
数据下的卷积操作, 并将3个通道上的值累加, 得到最终的卷积操作的结果.


#### 下采样层

在卷积层后面的下采样层, 主要是用一个特征来表达一个局部特征, 使得参数大为减少. 常见的有max-pooling、 mean-pooling、
L2-pooling.

#### 全连接层

实质为包含一个隐含层的神经网络模型. 全连接层的每一个节点都与上一层的所有节点相连, 用来把前边提取到的特征综合起来.

## 杂谈

- 重点: [梯度弥散与梯度爆炸](https://www.cnblogs.com/yangmang/p/7477802.html), 合理的使用激活函数
- [几种CNN算法](https://mp.weixin.qq.com/s?__biz=MzI5NTIxNTg0OA==&mid=2247485440&idx=1&sn=054105f9731120426f6b4c8ca17a4b6f&chksm=ec57bf87db203691cad15d122c2d3a3aac30d92e87ad5d60ce905e927f62a6bea62ca5238a26&mpshare=1&scene=23&srcid=0301egYFJGt5MHfhIQVWPuPp#rd)
    - 数据增强: 从原始图像中截取一定区域, 增加了指数倍的数据量.