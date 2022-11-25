#!/user/bin/python
# author jeff

from torch import nn
import torch.nn.functional as F


# 仅有两个全连接层
# 注意如果要用这个model,MNIST和fashionMnist数据的维度不用动
# 如果出现问题，检查一下数据处理那块
class Net2nn(nn.Module):
    def __init__(self):
        super(Net2nn, self).__init__()
        self.fc1 = nn.Linear(784, 200)  # 输入到隐藏层1
        self.fc2 = nn.Linear(200, 200)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(200, 10)  # 隐藏层2到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一次激活
        x = F.relu(self.fc2(x))  # 第二次激活
        x = self.fc3(x)
        return x
