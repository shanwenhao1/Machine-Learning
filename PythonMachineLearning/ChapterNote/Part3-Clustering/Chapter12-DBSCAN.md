# DBSCAN

基于密度的聚类算法, 可以发现任意形状的聚类.

通过在数据集中寻找被低密度区域分离的高密度区域, 将分离出的高密度区域作为一个独立的类别.

## 算法原理

DBSCAN(Density-Based Spatial Clustering of Application with Noise)是一种典型的基于密度的聚类算法. 有两个最基本
的领域参数, 分别为ε和MinPts(ε领域表示的是在数据集中与样本点x<sub>i</sub>的距离不大于ε的样本; MinPts表示的是
样本点x<sub>i</sub>的ε领域内的最少样本点的数目). MinPts即代表了样本密度. 

DBSCAN将数据点分为三类:
- 核心点(Core Points): 若样本x<sub>i</sub>的ε邻域内至少包含了MinPts个样本, 即|N<sub>ε</sub>(x<sub>i</sub>)| >= MinPts,
则称样本点x<sub>i</sub>为核心点.
- 边界点(Border Points): 若样本x<sub>i</sub>的ε邻域内包含的样本数目小于MinPts, 但是它在其他核心点的邻域内, 
则称样本点x<sub>i</sub>为边界点.
- 噪音点(Noise): 指的是既不是核心点也不是边界点的点.

密度可达概念:
- 直接密度可达(directly density-reachable): 若样本点x<sub>j</sub>在核心点x<sub>i</sub>的ε邻域内, 则称样本点x<sub>j</sub>
从样本点x<sub>i</sub>直接密度可达.
- 密度可达(density-reachable): 若在样本点x<sub>i,1</sub>和样本点x<sub>i,n</sub>之间存在序列
x<sub>i,2</sub>....,x<sub>i,n-1</sub>,且x<sub>i,j+1</sub>从x<sub>i,j</sub>直接密度可达, 则称
x<sub>i,n</sub>从x<sub>i,1</sub>密度可达.由密度可达的定义可知, 样本点x<sub>i,1</sub>,x<sub>i,2</sub>,....,
x<sub>i,n-1</sub>均为核心点, 直接密度可达是密度可达的特殊情况
- 密度连接(density-connected): 对于样本点x<sub>i</sub>和样本点x<sub>j</sub>, 若存在样本点x<sub>k</sub>, 使得
x<sub>i</sub>和x<sub>j</sub>都从x<sub>k</sub>密度可达, 则称x<sub>i</sub>和x<sub>j</sub>密度相连.

在DBSCAN算法中, 聚类"簇"定义为: 由密度可达关系导出的最大的密度连接样本的集合.

### 算法流程
R
- 根据给定的领域参数ε和MinPts确定所有的核心对象
- 对每一个核心对象
- 选择一个未处理过的核心对象, 找到由其密度可达的样本生成聚类"簇"
- 重复以上过程

### 参数选择

Eps的值可以使用绘制k-距离曲线(k-distance graph)方法得当,在k-距离曲线图明显拐点位置为对应较好的参数.
若参数设置过小,大部分数据不能聚类；若参数设置过大,多个簇和大部分对象会归并到同一个簇中.
- K-距离：K距离的定义在DBSCAN算法原文中给出了详细解说,给定K邻域参数k,对于数据中的每个点,计算对应的第k个
最近邻域距离,并将数据集所有点对应的最近邻域距离按照降序方式排序,称这幅图为排序的k距离图,选择该图中第一个
谷值点位置对应的k距离值设定为Eps.一般将k值设为4

MinPts的选取有一个指导性的原则（a rule of thumb）,MinPts≥dim+1,其中dim表示待聚类数据的维度.
MinPts设置为1是不合理的,因为设置为1,则每个独立点都是一个簇,MinPts≤2时,与层次距离最近邻域结果相同,
因此,MinPts必须选择大于等于3的值.若该值选取过小,则稀疏簇中结果由于密度小于MinPts,从而被认为是边界点
不被用于在类的进一步扩展；若该值过大,则密度较大的两个邻近簇可能被合并为同一簇.因此,该值是否设置适当会
对聚类结果造成较大影响.

## 杂谈

- [DBSCAN伪代码及原理](https://blog.csdn.net/xieruopeng/article/details/53675906)
- [DBSCAN优缺点及参数选择](https://blog.csdn.net/zhouxianen1987/article/details/68945844): 其中的链接比较全面.
