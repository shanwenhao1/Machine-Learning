# 个人学习的一些总结

## 分类

### 最小二乘法和最大似然估计的联系和区别

- 对于最小二乘法，当从模型总体随机抽取n组样本观测值后，最合理的参数估计量应该使得模型能最好地拟合样本数据，
也就是估计值和观测值之差的平方和最小。而对于最大似然法，当从模型总体随机抽取n组样本观测值后，最合理的参数估计
量应该使得从模型中抽取该n组样本观测值的概率最大
- 最小二乘法以估计值与观测值的差的平方和作为损失函数，极大似然法则是以最大化目标值的似然概率函数为目标函数，
从概率统计的角度处理线性回归并在似然概率函数为高斯函数的假设下同最小二乘建立了的联系。


## 回归

### 线性回归

- 输入数据是一些具有M维特征的向量, 然后回归模型的函数就是M维特征取值的线性加权, 这些权值就是我们要求取的参数.
- 定义损失函数: 比如最小二乘法使用模型的预测值和真实值的平方和作为损失函数. 有了损失函数, 参数求解的问题就成了
损失函数求和最小化的最优化问题
- 真实观测值 = 模型的预测值 + 残差, 最小二乘法有一个缺陷(如果残差不服从高斯分布, 则求解出来的模型就不正确)

#### 线性回归和岭回归和Lasso回归

- 都只能针对线性的数据
- 线性回归使用最小二乘法或者牛顿法作为损失函数, 但无法较好的处理特征之间关联性较高的回归问题
- 岭回归与Lasso回归都是添加正则项
    - 岭回归是在残差平方和基础上增加正则项(λw<sup>2</sup>)
    - Lasso回归是在残差平方和基础上增加正则项(λ|w|)
- 岭回归缺点及Lasso解决的问题, [详情请见](PythonMachineLearning/ChapterNote/Part2-Regression/Chapter8-RidgeAndLasso-Regression.md)

### 非线性回归
#### 局部加权线性回归
能够对非线性的数据实现较好拟合, 但是局部加权线性回归模型属于非参学习算法, 每次预测时, 
需要利用数据重新训练模型的参数(非参耗费时间).

#### [CART树回归](PythonMachineLearning/ChapterNote/Part2-Regression/Chapter9-CART-Regression.md)



### 什么是正则化

对损失函数(目标函数)加入一个惩罚项, 使得模型由多解变为更倾向其中一个解


## Clustering

- K-Means、K-Means++、Mean Shift算法都是基于距离的聚类算法, 聚类结果是球状的簇.
- DBSACN算法是基于密度的聚类算法, 可以发现任意形状的聚类

## 杂谈

- [岭回归和lasso回归](https://zm8.sm-tc.cn/?src=l4uLj4zF0NCIiIjRnJGdk5CYjNGckJLQjJeWh5aMl5qRmNCP0MjNys3HzsnRl4uSkw%3D%3D&uid=1d40bd6d5c3ab79707cdca47e5eec6e6&hid=ec0c6a268041c48b91eb571927a6d4f3&pos=1&cid=9&time=1528366715144&from=click&restype=1&pagetype=0020000000000408&bu=web&query=%E5%B2%AD%E5%9B%9E%E5%BD%92%E5%92%8Classo%E5%9B%9E%E5%BD%92&mode=&v=1&force=true&wap=false&province=%E7%A6%8F%E5%BB%BA%E7%9C%81&city=%E5%8E%A6%E9%97%A8%E5%B8%82&uc_param_str=dnntnwvepffrgibijbprsvdsdichei)