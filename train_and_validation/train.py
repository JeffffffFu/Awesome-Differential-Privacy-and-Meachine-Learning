import torch
from torch import nn
from torch.utils.data import TensorDataset
import pylab
import matplotlib.pyplot as plt


#MBSGD 小批量梯度下降 , 这个train_loader里面是有多个batch的
from train_and_validation.validation import validation
import torch.nn.functional as F


def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    train_acc= 0.0
    i=0

    for data, target in train_loader:  # batch之前组装到data数据集里的,pytorch的MBDG统一用这种方式进行,会按序列一个个btach训练

        output = model(data.to(torch.float32))  # 计算输出
        loss = F.cross_entropy(output, target.to(torch.long))  # 损失函数
        optimizer.zero_grad()  # 梯度清空
        loss.backward()  # 梯度求导
        optimizer.step()  # 参数优化更新



    return train_loss,train_acc
