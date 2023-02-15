import torch
from torch import nn

from model.CNN import CNN
from optimizer.dp_optimizer import DPAdam


def create_model_optimizer_criterion_dict(number_of_clients, learning_rate, model):
    clients_model_list = []
    clients_optimizer_list = []
    clients_criterion_list = []

    #为各个客户端分配model,optimizer等
    for i in range(number_of_clients):
        model_info = model
        # model_info = Net2nn()
        clients_model_list.append(model_info)

        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate)
        clients_optimizer_list.append(optimizer_info)

        criterion_info = nn.CrossEntropyLoss()
        clients_criterion_list.append(criterion_info)

    return clients_model_list, clients_optimizer_list, clients_criterion_list


#主要是在优化器上要绑定基于DP的优化器
def create_model_optimizer_criterion_dict_with_dp_optimizer(number_of_clients, learning_rate, momentum,max_norm,sigma,batch_size_of_each_clients):
    clients_model_list = []
    clients_optimizer_list = []
    clients_criterion_list = []

    #为各个客户端分配model,optimizer等
    for i in range(number_of_clients):
        model_info = CNN()
        # model_info = Net2nn()
        clients_model_list.append(model_info)

        optimizer_info = DPAdam(
            l2_norm_clip=max_norm,  # 裁剪范数
            noise_multiplier=sigma,
            minibatch_size=batch_size_of_each_clients[i], #batch_size
            microbatch_size=1,  # 几个样本梯度进行一次裁剪
            # 后面这些参数是继承父类的（SGD优化器的一些参数）
            params=model_info.parameters(),
            lr=learning_rate,
        )
        clients_optimizer_list.append(optimizer_info)

        criterion_info = nn.CrossEntropyLoss()
        clients_criterion_list.append(criterion_info)

    return clients_model_list, clients_optimizer_list, clients_criterion_list