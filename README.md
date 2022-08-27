# 说明
## utils.py
各种会使用到的函数
## train_cifar10.ipynb
使用正常数据训练模型，并保存checkpoint，并将训练时测试样本在每个epoch的表现情况存入train_detail文件夹中（yaml格式）
## tracin_e2.ipynb
根据checkpoint，通过部分测试数据（hard/easy）和训练数据计算tracin，并将训练数据排序后存放如tracin_detail_DLA文件夹中
## tracin_train
根据tracin_detail_DLA文件夹中的文件，选取相应的训练数据进行训练
