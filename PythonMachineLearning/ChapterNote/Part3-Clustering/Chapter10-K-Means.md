# K-Means算法

## 相似性的度量

一般定义一个距离函数d(X,Y), 利用d(X,Y)来表示样本X和样本Y之间的相似性.

通常使用的距离函数: (假设有两个点P和Q, 它们对应的坐标分别为: 
<br><center>![](../MularGif/Part3-Clustering/Chapter10Gif/PAndQ.gif)</center></br>)
- 闵可夫斯基距离(Minkowski Distance)
    - 点P和点Q之间的闵可夫斯基距离可以定义为:
    <br><center>![](../MularGif/Part3-Clustering/Chapter10Gif/Minkowski%20Distance.gif)</center></br>
- 曼哈顿距离(Manhattan Distance)
    - 点P和点Q之间的曼哈顿距离可以定义为:
    <br><center>![](../MularGif/Part3-Clustering/Chapter10Gif/Manhattan%20Distance.gif)</center></br>
- 欧氏距离(Euclidean Distance)
    - 点P和点Q之间的欧式距离可以定义为:
    <br><center>![](../MularGif/Part3-Clustering/Chapter10Gif/Euclidean%20Distance.gif)</center></br>
    
曼哈顿距离和欧氏距离都是闵可夫斯基距离的具体形式

## K-Means

### 算法原理

算法步骤:
- 初始化常数k,  随机初始化K个聚类中心
    - 缺点:
        - 需要事先知道有多少个类, 局限性较大
        - 随机初始化聚类中心, 对结果影响较大, 多次运行结果差异较大
        
- 重复计算以下过程, 直到聚类中心不再改变
    - 计算每个样本与每个聚类中心之间的相似度, 将样本划分到最相似的类别中
    - 计算划分到每个类别中的所有样本特征的均值, 并将该均值作为每个类新的聚类中心
- 输出最终的聚类中心以及每个样本所属的类别

## K-Means++

聚类中心的初始化过程的基本原则是使得聚类中心的距离尽可能原

## 算法原理

算法步骤:
- 在数据集中随机选择一个样本点作为第一个初始化的聚类中心:
- 选择出其余的聚类中心:
    - 计算样本中的每一个样本点与已经初始化的聚类中心之间的距离, 
    并选择其中最短的距离, 记为d<sub>1</sub>
    - 以概率选择距离最大的样本作为新的聚类中心, 重复上述过程, 直到k聚类中心都被确定
    - 对k个初始化的聚类中心, 利用K-Means算法计算吗最终的聚类中心

