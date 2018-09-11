# tensor基础教程

## MNIST教程

测试数据集存在的必要性: 提高泛化能力

MNIST中将图片展开成数组形式时, 会丢失掉二维结构信息. 这显然是不理想的,
因此我们还应思考多维向低纬展开时结构信息的提取, 考虑是否有价值.

[各种模型关于MNIST数据集预测性能对比表](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)