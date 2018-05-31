# Softmax Regression
它是Logistic Regression算法在多分类问题上的推广(其中, 任意两个类之间是线性可分的)

## 损失函数
类似于LR算法, 在Softmax Regression算法的损失函数引入指示函数I(·), 
<br><center>![具体形式](MularGif/Chapter2Gif/CostI.gif)</center></br>
<br>Softmax Regression的损失函数为:</br>
<br><center>![损失函数](MularGif/Chapter2Gif/LossMular.gif)</center></br>
<br>其中, ![](MularGif/Chapter2Gif/IYJ.gif)表示的是当y<sup>(i)</sup></br>属于第j类时, 
![](MularGif/Chapter2Gif/IYJ.gif) = 1, 否则 ![](MularGif/Chapter2Gif/IYJ.gif) = 0.