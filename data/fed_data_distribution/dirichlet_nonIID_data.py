import numpy as np
from torch.utils.data import DataLoader


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


def create_Non_iid_subsamples_dirichlet(n_clients, alpha, seed, data):
    """
    使用狄利克雷分布划分数据集
    x是数据，y是标签
    @Author:LingXinPeng
    """

    # 这里返回的是一个二维list，每个二级list装了对应下标的client分配到的数据的索引
    y_data=data.targets    #得到全部样本的标签

    client_idcs = dirichlet_split_noniid(train_labels=y_data, alpha=alpha, n_clients=n_clients, seed=seed)

    client_data_dict = dict()
    clients_data_list=[]


    for i in range(n_clients):
        indices = np.sort(client_idcs[i])
        data_info = DataLoader(data, sampler=indices)
        clients_data_list.append(data_info)

    return clients_data_list

def fed_dataset_NonIID_Dirichlet(train_data, n_clients, alpha, seed):
    """
    按Dirichlet分布划分Non-IID数据集，来源：https://zhuanlan.zhihu.com/p/468992765
    x是样本，y是标签
    :return:
    """
    # #调用create_Non_iid_subsamples_dirichlet拿到每个客户端的训练样本字典
    # client_data_dict = create_Non_iid_subsamples_dirichlet(n_clients=n_clients, alpha=alpha, seed=seed,data=train_data)
    # # 要把每个客户端的权重也返回去，后面做加权平均用
    # number_of_data_on_each_clients = [len(client_data_dict[key]) for key in client_data_dict.keys()]
    # total_data_length = sum(number_of_data_on_each_clients)
    # weight_of_each_clients = [x / total_data_length for x in number_of_data_on_each_clients]
    #
    #
    # print("··········让我康康y_trian_dict···········")
    # for key in client_data_dict.keys():
    #     print(key, len(client_data_dict[key]))
    #     lst = []
    #     for data, target in client_data_dict[key]:
    #         lst.append(target)
    #     for i in range(10):
    #         print(lst.count(i), end=' ')
    #     #print(len(client_data_dict[key].dataset.targets))
    #     print()
    # print("··········让我康康weight_of_each_clients···········")
    #
    # print(weight_of_each_clients) #权重打印


    #调用create_Non_iid_subsamples_dirichlet拿到每个客户端的训练样本字典
    clients_data_list = create_Non_iid_subsamples_dirichlet(n_clients=n_clients, alpha=alpha, seed=seed,data=train_data)
    # 要把每个客户端的权重也返回去，后面做加权平均用
    number_of_data_on_each_clients = [len(clients_data_list[i]) for i in range(len(clients_data_list))]
    total_data_length = sum(number_of_data_on_each_clients)
    weight_of_each_clients = [x / total_data_length for x in number_of_data_on_each_clients]


    print("··········让我康康y_trian_dict···········")
    for i in range(len(clients_data_list)):
        print(i, len(clients_data_list[i]))
        lst = []
        for data, target in clients_data_list[i]:
            lst.append(target)
        for i in range(10):     #0-9是标签，这个需要根据不同的数据集来打印，mnist和fashionmnist是只有0-9的标签
            print(lst.count(i), end=' ')
        #print(len(client_data_dict[key].dataset.targets))
        print()
    print("··········让我康康weight_of_each_clients···········")

    print(weight_of_each_clients) #权重打印


    return clients_data_list, weight_of_each_clients