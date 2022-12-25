import math

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


#《Federated Learning on Non-IID Data Silos: An Experimental Study》
#按Dirichlet分布划分Non-IID数据集：https://zhuanlan.zhihu.com/p/468992765
def dirichlet_split_noniid(train_labels, alpha, n_clients, seed):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    狄利克雷分布相关函数
    '''
    np.random.seed(seed)
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)  # 第一个参数是list，是n_clients个alpha
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]  # 替换成DataFrame
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            # enumerate在字典上是枚举、列举的意思
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    """这里返回的是一个二维list，每个二级list装了对应下标的client分配到的数据的索引"""
    return client_idcs


def create_Non_iid_subsamples_dirichlet(n_clients, alpha, seed, train_data):
    """
    使用狄利克雷分布划分数据集
    x是数据，y是标签
    @Author:LingXinPeng
    """

    # 这里返回的是一个二维list，每个二级list装了对应下标的client分配到的数据的索引
    train_labels=train_data.targets    #得到全部样本的标签

    client_idcs = dirichlet_split_noniid(train_labels, alpha, n_clients, seed)

    clients_data_list=[]



    for i in range(n_clients):
        indices = np.sort(client_idcs[i])
        indices=torch.tensor(indices)

        data=train_data.data / 255.0
        imgae=torch.index_select(data,0,indices)
        imgae=torch.unsqueeze(imgae,1)       #在1的位置增加一维

        targets=torch.index_select(train_labels,0,indices)
        data_info=TensorDataset(imgae,targets)
        clients_data_list.append(data_info)

    print("注意查看训练集的data的数据类型是不是0-255的无符号整型")
    #print("clients_data_list:",clients_data_list[1][1])
    return clients_data_list

def fed_dataset_NonIID_Dirichlet(train_data, n_clients, alpha, seed,q):
    """
    按Dirichlet分布划分Non-IID数据集，来源：https://zhuanlan.zhihu.com/p/468992765
    x是样本，y是标签
    :return:
    """

    #调用create_Non_iid_subsamples_dirichlet拿到每个客户端的训练样本字典
    clients_data_list = create_Non_iid_subsamples_dirichlet(n_clients, alpha, seed,train_data)
    # 要把每个客户端的权重也返回去，后面做加权平均用
    number_of_data_on_each_clients = [len(clients_data_list[i]) for i in range(len(clients_data_list))]
    total_data_length = sum(number_of_data_on_each_clients)
    weight_of_each_clients = [x / total_data_length for x in number_of_data_on_each_clients]


    print("··········让我康康y_trian_dict···········")
    for i in range(len(clients_data_list)):
        print(i, len(clients_data_list[i]))
        lst = []
        for data, target in clients_data_list[i]:
            #print("target:",target)
            lst.append(target.item())

        for i in range(10):     #0-9是标签，这个需要根据不同的数据集来打印，mnist和fashionmnist是只有0-9的标签
            print(lst.count(i), end=' ')
        #print(len(client_data_dict[key].dataset.targets))
        print()
    print("··········让我康康weight_of_each_clients···········")

    print(weight_of_each_clients) #权重打印

    batch_size_of_each_clients=[ math.floor(len(clients_data_list[i]) * q) for i in range(len(clients_data_list))]

    return clients_data_list, weight_of_each_clients,batch_size_of_each_clients