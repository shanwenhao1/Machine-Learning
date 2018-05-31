# Deep Learning Tutorial

## DeepLearning 介绍
机器学习本质上就是寻找一个判断方法(Machine Learning ≈ Looking for a Function)

监督学习的FrameWork(基本步骤): 预设一系列的方法 → 找出最好好的方法 ← 训练数据,  最后再用测试集测试方法的准确性.


### Deep Learning的基本思想 [摘至浅谈深度学习(Deep Learning)的基本思想和方法](https://blog.csdn.net/xianlingmao/article/details/8478562)
假设我们有一个系统S,它有n层(S1,...Sn),它的输入是I,输出是O,形象地表示为： I =>S1=>S2=>.....=>Sn => O,
如果输出O等于输入I,即输入I经过这个系统变化之后没有任何的信息损失,保持了不变,这意味着输入I经过每一层Si都没有
任何的信息损失,即在任何一层Si,它都是原有信息(即输入I)的另外一种表示.现在回到我们的主题Deep Learning,我们需要
自动地学习特征,假设我们有一堆输入I(如一堆图像或者文本),假设我们设计了一个系统S(有n层),我们通过调整系统中参数,
使得它的输出仍然是输入I,那么我们就可以自动地获取得到输入I的一系列层次特征,即S1,..., Sn.

另外,前面是假设输出严格地等于输入,这个限制太严格,我们可以略微地放松这个限制,例如我们只要使得输入与输出的差别尽
可能地小即可,这个放松会导致另外一类不同的Deep Learning方法.上述就是Deep Learning的基本思想.

### Deep Learning的常用方法

#### AutoEncoder
最简单的一种方法是利用人工神经网络的特点,人工神经网络(ANN)本身就是具有层次结构的系统,如果给定一个神经网络,
我们假设其输出与输入是相同的,然后训练调整其参数,得到每一层中的权重,自然地,我们就得到了输入I的几种不同表示
(每一层代表一种表示),这些表示就是特征,在研究中可以发现,如果在原有的特征中加入这些自动学习得到的特征可以大大
提高精确度,甚至在分类问题中比目前最好的分类算法效果还要好！这种方法称为AutoEncoder.当然,我们还可以继续加上一些
约束条件得到新的Deep Learning方法,如如果在AutoEncoder的基础上加上L1的Regularity限制(L1主要是约束每一层中的节点中
大部分都要为0,只有少数不为0,这就是Sparse名字的来源),我们就可以得到Sparse AutoEncoder方法.

Sparse Coding
如果我们把输出必须和输入相等的限制放松,同时利用线性代数中基的概念,即O = w1*B1 + W2*B2+....+ Wn*Bn, 
Bi是基,Wi是系数,我们可以得到这样一个优化问题：
                                                              Min |I - O| 
通过求解这个最优化式子,我们可以求得系数Wi和基Bi,这些系数和基础就是输入的另外一种近似表达,因此,它们可以特征来
表达输入I,这个过程也是自动学习得到的.如果我们在上述式子上加上L1的Regularity限制,得到：
                                                              Min |I - O| + u*(|W1| + |W2| + ... + |Wn|)

这种方法被称为Sparse Coding.

#### Restrict Boltzmann Machine(RBM)
假设有一个二部图,每一层的节点之间没有链接,一层是可视层,即输入数据层(v),一层是隐藏层(h),如果假设所有的节点
都是二值变量节点(只能取0或者1值),同时假设全概率分布p(v, h)满足Boltzmann 分布,我们称这个模型是R
estrict  Boltzmann Machine (RBM).下面我们来看看为什么它是Deep Learning方法.首先,这个模型因为是二部图,所以在
已知v的情况下,所有的隐藏节点之间是条件独立的,即p(h|v) =p(h1|v).....p(hn|v).同理,在已知隐藏层h的情况下,所有的
可视节点都是条件独立的,同时又由于所有的v和h满足Boltzmann 分布,因此,当输入v的时候,通过p(h|v) 可以得到隐藏层h,
而得到隐藏层h之后,通过p(v|h) 又能得到可视层,通过调整参数,我们就是要使得从隐藏层得到的可视层v1与原来的可视层v如果
一样,那么得到的隐藏层就是可视层另外一种表达,因此隐藏层可以作为可视层输入数据的特征,所以它就是一种Deep Learning方法.

如果,我们把隐藏层的层数增加,我们可以得到Deep Boltzmann Machine (DBM);如果我们在靠近可视层的部分使用贝叶斯信念网络
(即有向图模型,当然这里依然限制层中节点之间没有链接),而在最远离可视层的部分使用Restrict  Boltzmann Machine,我们
可以得到Deep Belief Net (DBN) .


### 章节笔记
- [LECTURE1 Introduction](/DeepLearningTutorial/ChapterNote/Introduction.md),
[他人阅读笔记](https://blog.csdn.net/qq_33120943/article/details/78487791)
- [Tips for Training DNN](/DeepLearningTutorial/ChapterNote/TipsOfTrainingDNN.md)
    
### [杂谈](/DeepLearningTutorial/Terminology.md)


