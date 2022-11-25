import numpy as np


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
    y_data=data.target

    client_idcs = dirichlet_split_noniid(train_labels=y_data, alpha=alpha, n_clients=n_clients, seed=seed)

    client_data = dict()


    for i in range(n_clients):
        client_name=str(i)
        indices = np.sort(client_idcs[i])
        data_info = data[indices, :]
        client_data.update({client_name: data_info})


    return client_data




def fed_dataset_NonIID_Dirichlet(train_data, n_clients, alpha, seed):
    """
    按Dirichlet分布划分Non-IID数据集，来源：https://zhuanlan.zhihu.com/p/468992765
    x是样本，y是标签
    :return:
    """
    client_data = create_Non_iid_subsamples_dirichlet(n_clients=n_clients, alpha=alpha, seed=seed,data=train_data)
    # 要把每个客户端的权重也返回去，后面做加权平均用
    number_of_data_on_each_clients = [len(y_train_dict[key]) for key in y_train_dict.keys()]
    total_data_length = sum(number_of_data_on_each_clients)
    weight_of_each_clients = [x / total_data_length for x in number_of_data_on_each_clients]

    print("··········让我康康y_trian_dict···········")
    lst = []
    for key in y_train_dict.keys():
        print(key, len(y_train_dict[key]))
        lst = []
        for i in range(len(y_train_dict[key])):
            lst.append(int(y_train_dict[key][i]))
        for i in range(10):
            print(lst.count(i), end=' ')
        print()
    print("··········让我康康weight_of_each_clients···········")

    print(weight_of_each_clients) #权重打印


    return x_train_dict, y_train_dict, weight_of_each_clients