



# 该函数在各个节点中训练本地模型,可打印部分客户端的训练信息,这个是无裁剪的梯度下降
import math

import torch
from torch.utils.data import TensorDataset

from train_and_validation.train import train


def local_clients_train_process_without_dp(number_of_clients,clients_data_list,clients_model_list,clients_criterion_list,clients_optimizer_list,numEpoch,q):

    # 循环客户端
    for i in range(number_of_clients):

        batch_size=math.floor(len(clients_data_list[i])*q)
        train_dl = torch.utils.data.DataLoader(
            clients_data_list[i], batch_size=batch_size, shuffle=False,drop_last=True).dataset  #这里多加了个.dataset，因为之前做迪利克雷的时候做了一次dataloader
        # 各客户端取到自己对应的模型,损失函数和优化器
        model = clients_model_list[i]
        criterion = clients_criterion_list[i]
        optimizer = clients_optimizer_list[i]

        if i < number_of_clients:
            print("Client:", i)

        for epoch in range(numEpoch):  # 每个客户端本地进行训练


            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer)  # 直接CNN->net2nn这里有问题
            #test_loss, test_accuracy = validation(model, test_dl)  联邦下，这里本地没有合适的测试集了

            if i < number_of_clients:
                print("epoch: {:3.0f}".format(epoch + 1) + " | train_loss: {:7.5f}".format(
                    train_loss) + " | train_accuracy: {:7.5f}".format(train_accuracy))


