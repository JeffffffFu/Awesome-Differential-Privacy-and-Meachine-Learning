import torch
from torch.utils.data import TensorDataset
import numpy as np
import torch.nn.functional as F


# 这里实际上做的也是小批量梯度下降，在裁剪上是逐样本，为了更好的控制DPSGD，这边的train_loader我们希望只有一个batch就好，多个也可以
def train_dynamic_add_noise(model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    aa = 0
    train_acc = 0.
    i = 0
    for id, (data, target) in enumerate(train_loader):
        optimizer.zero_accum_grad()  # 梯度清空
        for iid, (X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):  # 这里相当于逐样本
            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch.to(torch.float32), 0))  # 这要是这里要做升维

            loss = F.cross_entropy(output, torch.unsqueeze(y_microbatch.to(torch.long), 0))

            loss.backward()  # 梯度求导，这边求出梯度
            optimizer.microbatch_step()  # 这个step做的是每个样本的梯度裁剪和梯度累加的操作
        optimizer.step_dp()  # 这个做的是梯度加噪和梯度平均更新下降的操作

        # 训练集测试损失值和准确率
        # train_output=model(data.to(torch.float32))
        # train_loss=criterion(train_output,target).item()
        # prediction = train_output.argmax(dim=1, keepdim=True)  # 将one-hot输出转为单个标量
        # correct = prediction.eq(target.view_as(prediction)).sum().item()  # 比较得到准确率
        # train_acc=100. * correct/len(data)
        # i+=1

        # print(f'batch: {i}, 'f'Train set: loss: {train_loss:.4f}, '
        #       f'Accuracy: {correct}/{len(data)} ({train_acc:.2f}%)')

    return train_loss, train_acc  # 返回平均损失和平均准确率


def train_dynamic_add_noise_geo_full_batch(model, train_loader, optimizer):
    '''
    Args:
        model:
        train_loader: PyG DataLoader
        optimizer:

    Returns:

    '''
    model.train()
    train_loss = 0.0
    aa = 0
    train_acc = 0.
    i = 0
    for id, data in enumerate(train_loader):  # TODO per-sample computation for mini-batch
        # X_microbatch = data
        # y_microbatch = data.y
        optimizer.zero_accum_grad()  # 梯度清空

        optimizer.zero_microbatch_grad()
        output = model(data)
        # output = model(torch.unsqueeze(X_microbatch.to(torch.float32), 0))  # 这要是这里要做升维

        # loss = F.cross_entropy(output, torch.unsqueeze(y_microbatch.to(torch.long), 0))
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()  # 梯度求导，这边求出梯度
        optimizer.microbatch_step()  # 这个step做的是每个样本的梯度裁剪和梯度累加的操作
        optimizer.step_dp()  # 这个做的是梯度加噪和梯度平均更新下降的操作
        train_loss += loss
    return train_loss, train_acc  # 返回平均损失和平均准确率


def train_dynamic_add_noise_geo_mini_batch(model, train_loader, optimizer):
    '''
    Args:
        model:
        train_loader: PyG DataLoader
        optimizer:

    Returns:

    '''
    model.train()
    train_loss = 0.0
    aa = 0
    train_acc = 0.
    i = 0
    criterion = torch.nn.CrossEntropyLoss()

    for id, data in enumerate(train_loader):  # TODO per-sample computation for mini-batch
        optimizer.zero_accum_grad()  # 梯度清空
        for id in range(data.num_graphs):
            data_microbatch = data[id]
            optimizer.zero_microbatch_grad()
            # out = model(data_microbatch.x, data_microbatch.edge_index, data_microbatch.batch)
            out = model(data_microbatch.x, data_microbatch.edge_index, torch.tensor([0], dtype=torch.int64))
            loss = criterion(out, data_microbatch.y)
            loss.backward()  # 梯度求导，这边求出梯度
            optimizer.microbatch_step()  # 这个step做的是每个样本的梯度裁剪和梯度累加的操作
        optimizer.step_dp()  # 这个做的是梯度加噪和梯度平均更新下降的操作
        train_loss += loss
    return train_loss, train_acc  # 返回平均损失和平均准确率


def train_dynamic_add_noise_geo_uniform_batch(model, train_loader, optimizer, criterion, full_batch=False):
    '''
    Args:
        model:
        train_loader: PyG DataLoader
        optimizer:

    Returns:

    '''
    model.train()
    train_loss = 0.0
    aa = 0
    train_acc = 0.
    i = 0

    for id, data in enumerate(train_loader):  # TODO per-sample computation for mini-batch
        optimizer.zero_accum_grad()  # 梯度清空
        if not full_batch: # pre-sample计算
            for id in range(data.num_graphs):
                data_microbatch = data[id]
                optimizer.zero_microbatch_grad()
                # out = model(data_microbatch.x, data_microbatch.edge_index, data_microbatch.batch)
                out = model(data_microbatch)
                loss = criterion(out, data_microbatch.y)
                loss.backward()  # 梯度求导，这边求出梯度
                optimizer.microbatch_step()  # 这个step做的是每个样本的梯度裁剪和梯度累加的操作
                train_loss += loss.item()
        else:
            optimizer.zero_microbatch_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()  # 梯度求导，这边求出梯度
            optimizer.microbatch_step()  # 这个step做的是每个样本的梯度裁剪和梯度累加的操作
            train_loss += loss.item()
        optimizer.step_dp()  # 这个做的是梯度加噪和梯度平均更新下降的操作
    return train_loss, train_acc  # 返回平均损失和平均准确率


def train_dynamic_add_noise_split_vector(model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    aa = 0
    train_acc = 0.
    i = 0
    for id, (data, target) in enumerate(train_loader):
        optimizer.zero_accum_grad()  # 梯度清空
        for iid, (X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):  # 这里相当于逐样本
            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch.to(torch.float32), 0))  # 这要是这里要做升维

            loss = F.cross_entropy(output, torch.unsqueeze(y_microbatch.to(torch.long), 0))

            loss.backward()  # 梯度求导，这边求出梯度
            optimizer.microbatch_step()  # 这个step做的是每个样本的梯度裁剪和梯度累加的操作
        optimizer.step_dp_split_vector()  # 这个做的是梯度加噪和梯度平均更新下降的操作

        # 训练集测试损失值和准确率
        # train_output=model(data.to(torch.float32))
        # train_loss=criterion(train_output,target).item()
        # prediction = train_output.argmax(dim=1, keepdim=True)  # 将one-hot输出转为单个标量
        # correct = prediction.eq(target.view_as(prediction)).sum().item()  # 比较得到准确率
        # train_acc=100. * correct/len(data)
        # i+=1

        # print(f'batch: {i}, 'f'Train set: loss: {train_loss:.4f}, '
        #       f'Accuracy: {correct}/{len(data)} ({train_acc:.2f}%)')

    return train_loss, train_acc  # 返回平均损失和平均准确率
