# TensorFlow Document Note

## TensorFlow设计 

### GPU and CPU running

TensorFlow将图形定义转换成分布式执行的操作, 不需要显示
指定使用CPU或GPU(TensorFlow自动检测, 尽量利用找到的第一个GPU来执行操作).

如果有多个GPU存在, 则除了第一个GPU外, 其余GPU默认不参与计算, 为了有效利用这些
GPU, 需将operation(节点, 简称op)指派给它们执行.
```python
import tensorflow as tf

with tf.Session() as sess:
    # "/cpu: 0": 机器的CPU, "/cpu: n": 机器的第n+1个GPU
    with tf.device("/gpu: 1"):
        matrix1 = tf.constant([[3., 3.]])
        ...
```

### Tensor

TensorFlow中使用tensor数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是tensor.
你可以把tensor看作是一个n维的数组或列表.

### feed

使用feed作为run的参数, 只在调用它的方法内有效, 方法结束, feed就消失.

## 杂谈

&nbsp;&nbsp;&nbsp;由于Numpy内类似于矩阵乘法这样的复杂运算是使用其他语言实现的, 这样就存在一个切换的开销问题,
而TensorFlow设计的思想是用图描述一系列的可交互的计算操作, 然后一起使用外部语言进行运算, 
从而减少切换的开销.
