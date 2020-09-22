# 使用TensorRT做encode
官方有个fast transformer
这个项目demo/Bert下改的

## 模型转换成.engine文件，在bert_build文件下做如下修改
1. 删除了squad函数
2. 删除了squad的调用

## 修改后再改data_processing文件
1. 原来是处理MC数据的，现在改成处理一般句子的


## 修改BERT_TRT
1. 原来是处理MC数据的，现在改成处理一般句子的



OK，完成以上操作，TensorRT的bert就完成了