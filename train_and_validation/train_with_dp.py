import torch
from torch.utils.data import TensorDataset
import numpy as np


#这里实际上做的也是小批量梯度下降，在裁剪上是逐样本，为了更好的控制DPSGD，这边的train_loader我们希望只有一个batch就好
def train_dynamic_add_noise(model, train_loader, criterion, optimizer):
    model.train()
    microbatch_loss=0.0
    train_loss = 0.0
    correct = 0
    group=0
    for id,(data, target) in enumerate(train_loader):  # batch之前组装到data数据集里的,pytorch的MBDG统一用这种方式进行,会按序列一个个batch训练
        #print("id:",id)
        optimizer.zero_accum_grad()  # 梯度清空

        #这里执行单样本操作，但是没有参数决定是单样本，依赖这里面的数据集的组装形式（TensorDataset(data, target)），和上面的train_loader一样，默认都是一个torch一个torch来
        # 数据组装中torch的维度决定你想要进行多少样本的梯度训练，取决于一开始的数据组装的结构
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):  #这里相当于逐样本

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch.to(torch.float32), 0))    #这要是这里要做升维

            loss = criterion(output, torch.unsqueeze(y_microbatch.to(torch.long), 0))       #相反，这边对于的output就不用升维了

            loss.backward()         #梯度求导，这边求出梯度
            optimizer.microbatch_step()  # 这个step做的是每个样本的梯度裁剪和梯度累加的操作

            train_loss += loss.item()  # 损失累加
            prediction = output.argmax(dim=1, keepdim=True)  # 将one-hot输出转为单个标量
            correct += prediction.eq(y_microbatch.view_as(prediction)).sum().item()  # 比较得到准确率


        optimizer.step()  # 这个做的是梯度加噪和梯度平均更新下降的操作


        # print(f'Train set: Average loss: {train_loss/ len(data):.4f}, '
        #       f'Accuracy: {correct}/{len(data)} ({correct/len(data):.2f}%)')

    return train_loss / len(data), correct / len(data)  # 返回平均损失和平均准确率