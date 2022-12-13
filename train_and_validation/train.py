import torch
from torch import nn
from torch.utils.data import TensorDataset


#MBSGD 小批量梯度下降 , 这个train_loader里面是有多个batch的
def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    i=0

    for data, target in train_loader:  # batch之前组装到data数据集里的,pytorch的MBDG统一用这种方式进行,会按序列一个个btach训练
        # print("data:",data.shape)
        output = model(data.to(torch.float32))  # 计算输出
        loss = criterion(output, target.to(torch.long))  # 损失函数
        optimizer.zero_grad()  # 梯度清空
        loss.backward()  # 梯度求导
        optimizer.step()  # 参数优化更新


        #训练集测试
        train_output = model(data.to(torch.float32))  # 计算输出
        train_loss = criterion(train_output, target.to(torch.long)).item()  # 损失函数
        prediction = train_output.argmax(dim=1, keepdim=True)  # 将one-hot输出转为单个标量
        correct = prediction.eq(target.view_as(prediction)).sum().item()  # 比较得到准确率
        train_acc=100. * correct/len(data)
        i+=1

        # print(f'batch: {i}, 'f'Train set: loss: {train_loss:.4f}, '
        #       f'Accuracy: {correct}/{len(data)} ({train_acc:.2f}%)')


    return train_loss,train_acc
