# DBSCAN

基于密度的聚类算法, 可以发现任意形状的聚类.

通过在数据集中寻找被低密度区域分离的高密度区域, 将分离出的高密度区域作为一个独立的类别.

## 算法原理

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

- 根据给定的领域参数ε和MinPts确定所有的核心对象
- 对每一个核心对象
- 选择一个未处理过的核心对象, 找到由其密度可达的样本生成聚类"簇"
- 重复以上过程


## 杂谈

- [DBSCAN伪代码及原理](https://blog.csdn.net/xieruopeng/article/details/53675906)
