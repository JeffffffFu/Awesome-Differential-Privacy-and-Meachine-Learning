import torch
from torch import nn
from torch.utils.data import TensorDataset


#MBSGD 小批量梯度下降
def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:  # batch之前组装到data数据集里的,pytorch的MBDG统一用这种方式进行,会按序列一个个btach训练

        output = model(data.to(torch.float32))  # 计算输出
        loss = criterion(output, target.to(torch.long))  # 损失函数

        optimizer.zero_grad()  # 梯度情况
        loss.backward()  # 梯度求导

        optimizer.step()  # 参数优化更新

    train_loss = loss.item()  # 损失累加
    prediction = output.argmax(dim=1, keepdim=True)  # 将one-hot输出转为单个标量
    correct = prediction.eq(target.view_as(prediction)).sum().item()  # 比较得到准确率

    return train_loss,correct
    # return train_loss / len(train_loader), correct / len(train_loader.dataset)  # 返回平均损失和平均准确率