# LECTURE1 Introduction

## 全连接前馈神经网络
- 介绍
    - 多个 [单个神经元](/DeepLearningTutorial/Pictures/Neuron.png)构成 
    [前馈神经网络](/DeepLearningTutorial/Pictures/Fully%20Connect%20Feedforward%20Network.png) 
    → 前馈神经网络构建[机器学习中间层](/DeepLearningTutorial/Pictures/Hidden%20Layer.png) 
    → [结果](/DeepLearningTutorial/Pictures/Neuron%20Result.png) 
    ([使用示例](/DeepLearningTutorial/Pictures/Example%20Judge%20Two.png))
        - 神经网络的逻辑单元: 输入向量x(input layer), 中间层a(2, i)(hidden layer), 输出层h(x)(output layer)
        - 前馈神经网络是一种最简单的神经网络, 各神经元分层排列. 每个神经元只与前一层的审计员相连, 接受前一层的输出
        并输出给下一层, 各层间没有反馈(可用一个有向无环图表示). 
        [详解](https://baike.baidu.com/item/%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/7580523?fr=aladdin)
        - 问题
            - 中间层需要多少层？ 不断试验(Trial and Error) + 直觉(Intuition)
    
- 实现
    - Training Data: 带有标签的数据集合
        - 评判标准Total Loss: 利用所有训练集数据寻找到Total Loss最小的方法 → 再寻找到Total Loss最小的神经网络参数θ
    
    - How to pick the best function
        - 找最优模型等效于得出一系列模型参数(weights & b)使得总体误差最小化.
        - 梯度下降(Gradient Descent)
            - step1: 首先假设一个模型, 具有模型参数{w1, w2,...., b1, b2,...}, 这个模型的总体误差为L, 梯度下降算法将对
            单个参数w进行处理.
            - step2: 给w定义一个初始值, 这个初始值可以是个随机值, 也可以是一个
            [RBM](https://blog.csdn.net/tsb831211/article/details/52757261)值.但要注意梯度下降不保证全局最小值, 
            不同的起点值会到达不同的最小值点.
            - [step3](/DeepLearningTutorial/Pictures/Learning%20Rate.png): 对总体误差L求w的偏导, 如果偏导值为正数, 
            则减小w, 如果偏导值为复数, 则增大w, 因此我们定义一个公式, 以便求解下一个w值.
            - step4: 对所有的模型参数进行梯度下降的处理后可以得到一个最终的坐标(w1, w2,...., b1, b3,....), 
            [如图](/DeepLearningTutorial/Pictures/Best%20Module.png), 就能得到一个最佳的模型.
            
    - [公式推导视频](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/DNN%20backprop.ecm.mp4/index.html) 
        - Backpropagation思想
            
- Why Deep?
    - more parameters, better performance
    - 采用Deep neural network 而不使用 Fat neural network 的原因
        - Deep方式可以用更少的数据完成分类, [图解](/DeepLearningTutorial/Pictures/Why%20Deep.png)
    


