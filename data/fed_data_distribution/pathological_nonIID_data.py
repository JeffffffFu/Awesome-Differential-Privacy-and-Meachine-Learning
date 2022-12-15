

import math

import numpy as np
import torch
from torch.utils.data import TensorDataset

# 病态的数据分布来源《Communication-efficient learning of deep networks from decentralized data》
# 联邦学习：按病态非独立同分布划分Non-IID样本：https://zhuanlan.zhihu.com/p/478981613
#60000个样本十个客户端，一个客户端6000个样本,一个样本大概4个切片
def pathological_split_noniid(train_data, n_clients, alpha, seed,q):
    # prepare training data
    # data_file = "../data/mnist/MNIST/processed/training.pt"
    # data, targets = torch.load(data_file)

    y_data = train_data.targets  # 得到全部样本的标签
    ind = np.argsort(y_data)  # 按照y_train标签进行排序，形成一个数组，经过排序后的数组的index值， [30207  5662 55366 ... 23285 15728 11924]

    # train_data.data里面是0-255的整型，和test数据集里的0-1的浮点型不一样，要统一做归一化处理
    x_data = train_data.data / 255.0

    # 用上面的Index，进行data和targets的排序
    x_data = x_data[ind]
    y_data = y_data[ind]

    # print("train_data[0]:",train_data[0])
    #
    # #为什么一train_data.__dict__之后，就变成整型了
    # print("train_data:",train_data.__dict__)
    #
    # ##问题出现在这个数据集划分这里，为什么本来


    shards = np.arange(40)  # 切片数40，10个客户端一个客户端4个切片，一个切片1500个样本，[0 1 2 .... 399] 从0到400的数组
    shard_size = len(y_data) / 40  # 每个切片的大小
    shard_per_client = int(40 / n_clients)  # 每个客户端应该得到的切片数
    np.random.seed(seed)
    np.random.shuffle(shards)  # 对切片进行随机扰动，这里要不要设置随机种子

    # prepare training data
    train_labels = []
    clients_data_list=[]
    print("注意查看训练集的data的数据类型是不是0-255的无符号整型")
    # print("x_data:",x_data[1])

    for ii in range(n_clients):  # 循环客户端
        my_shards = shards[(ii * shard_per_client): (ii + 1) * shard_per_client]  # 每个客户端分配四个shards，0-4，4-8，8-12... shards已经打乱了，前面0-4只是序号，对应的数字可能是50，48，02，160，这个才是切片序号
        ind = np.array([]).astype(int)
        for jj in my_shards:  # 分别循环切片序号
            ind = np.append(
                arr=ind, values=np.arange((jj * shard_size), (jj + 1) * shard_size)  # 这里利用打乱的切片序号得到对应的排序好后打乱的data的index
            )

        x_data_info=x_data[ind]
        x_data_info = torch.unsqueeze(x_data_info, 1)
        y_data_info=y_data[ind]


        data_info=TensorDataset(x_data_info,y_data_info)
        clients_data_list.append(data_info)

        #这边客户端数量一致，如果按照客户端数量进行加权就直接分权重和batch了
        weight_of_each_clients = [(1/n_clients) for x in range(n_clients)]
        batch_size_of_each_clients=[math.floor(q*len(clients_data_list[0])) for x in range(n_clients)]

    print("··········让我康康y_trian_dict···········")
    for i in range(len(clients_data_list)):
        print(i, len(clients_data_list[i]))
        lst = []
        for data, target in clients_data_list[i]:
            # print("target:",target)
            lst.append(target.item())

        for i in range(10):  # 0-9是标签，这个需要根据不同的数据集来打印，mnist和fashionmnist是只有0-9的标签
            print(lst.count(i), end=' ')
        # print(len(client_data_dict[key].dataset.targets))
        print()
    print("··········让我康康weight_of_each_clients···········")
    print("weight_of_each_clients:",weight_of_each_clients)
    print("batch_size_of_each_clients:",batch_size_of_each_clients)

    return clients_data_list,weight_of_each_clients,batch_size_of_each_clients



