import torch
from torch import nn

from model.CNN import CNN


def create_model_optimizer_criterion_dict(number_of_clients, learning_rate, momentum):
    clients_model_list = []
    clients_optimizer_list = []
    clients_criterion_list = []

    #为各个客户端分配model,optimizer等
    for i in range(number_of_clients):
        model_info = CNN()
        # model_info = Net2nn()
        clients_model_list.append(model_info)

        optimizer_info = torch.optim.Adam(model_info.parameters(), lr=learning_rate)

        clients_optimizer_list.append(optimizer_info)

        criterion_info = nn.CrossEntropyLoss()
        clients_criterion_list.append(criterion_info)

    return clients_model_list, clients_optimizer_list, clients_criterion_list