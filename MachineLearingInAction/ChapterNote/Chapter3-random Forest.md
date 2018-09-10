# 决策树

## 决策树的构造

优点: 计算复杂度不高, 输出结果易于理解, 对中间值的缺失不敏感, 可以处理不相关特征数据.

缺点: 可能会产生过拟合问题(剪枝的必要性)

构造树的关键在于划分子树时, 最优划分特征的选取. 按照此原则划分下去 + 剪枝就能划分出最佳决策树.

衡量选取最佳的特征有如下方法, [详情请见](../../PythonMachineLearning/ChapterNote/Part1-Classification/Chapter5-Random-Forest.md)): 
- 信息增益:
    - 香农信息熵的引入
- Gini impurity(基尼不纯度): 从一个数据集中随机选取子项, 度量其被错误分类到其他分组里的概率.


### 决策树的一般流程

- 收集数据
- 准备数据: 树构造法只适用于标称型数据, 因此数值型数据必须离散化
- 分析数据: 构造树完成之后, 应检查图形是否符合预期
- 训练算法: 构造树的数据结构
- 测试算法: 使用经验树计算错误率
- 使用算法: 此步骤可以适用于任何监督式学习算法, 而且使用决策树可以更好的理解数据的内在含义.