

import math

import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader


# 病态的数据分布来源《Communication-efficient learning of deep networks from decentralized data》
# 联邦学习：按病态非独立同分布划分Non-IID样本：https://zhuanlan.zhihu.com/p/478981613
#60000个样本十个客户端，一个客户端6000个样本,一个样本大概4个切片
from data.util.custom_tensor_dataset import CustomTensorDataset

#按标签排序，然后划分成40个切片，然后每个客户端随机分配几个切片。适用于十个客户端的。
def pathological_split_noniid(train_data, n_clients, alpha, seed,q):

    num_shards=4*n_clients  #默认一个得到4个切片
    if train_data.data.ndim==4:  #默认这个是cifar10,下面的transforms参数来源于getdata时候的参数
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    else:  #这个是mnist和fmnist数据
        train_data.data = torch.unsqueeze(train_data.data, 3)  #升维为NHWC，默认1通道。这边注意我们不需要转换维度，CustomTensorDataset包装后，后面会自动转换维度
        transform = torchvision.transforms.ToTensor()

    x_data = torch.Tensor(train_data.data)
    y_data = torch.Tensor(train_data.targets)  # 得到全部样本的标签


    # 进行data和targets的排序
    ind=np.argsort(y_data)
    x_data = x_data[ind]
    y_data = y_data[ind]


    shards = np.arange(num_shards)  # 切片数40，10个客户端一个客户端4个切片，一个切片1500个样本，[0 1 2 .... 399] 从0到400的数组
    shard_size = len(y_data) / num_shards  # 每个切片的大小
    shard_per_client = int(num_shards / n_clients)  # 每个客户端应该得到的切片数
    np.random.seed(seed)
    np.random.shuffle(shards)  # 对切片进行随机扰动

    # prepare training data
    train_labels = []
    clients_data_list=[]
    # print("x_data:",x_data[1])

    for ii in range(n_clients):  # 循环客户端
        my_shards = shards[(ii * shard_per_client): (ii + 1) * shard_per_client]  # 每个客户端分配四个shards，0-4，4-8，8-12... shards已经打乱了，前面0-4只是序号，对应的数字可能是50，48，02，160，这个才是切片序号
        ind = np.array([]).astype(int)
        for jj in my_shards:  # 分别循环切片序号
            ind = np.append(
                arr=ind, values=np.arange((jj * shard_size), (jj + 1) * shard_size)  # 这里利用打乱的切片序号得到对应的排序好后打乱的data的index
            )

        x_data_info=x_data[ind]
        y_data_info=y_data[ind]

        data_info=CustomTensorDataset((x_data_info,y_data_info),transform)
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


#从训练集中划分出部分验证集
def dividing_validation_set(train_data,validation_num):

    if train_data.data.ndim==4:  #默认这个是cifar10,下面的transforms参数来源于getdata时候的参数
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    else:  #这个是mnist和fmnist数据
        train_data.data = torch.unsqueeze(train_data.data, 3)  #升维为NHWC，默认1通道。这边注意我们不需要转换维度，CustomTensorDataset包装后，后面会自动转换维度
        transform = torchvision.transforms.ToTensor()

    x_data = torch.Tensor(train_data.data)
    y_data = torch.Tensor(train_data.targets)  # 得到全部样本的标签

    ind1=[]
    ind2=[]
    for i in range(len(x_data)-validation_num):
        ind1.append(i)
    for i in range(len(x_data)-validation_num,len(x_data)):
        ind2.append(i)

    x_data_info1 = x_data[ind1]
    y_data_info1 = y_data[ind1]
    train_tensor_dataset = CustomTensorDataset((x_data_info1,y_data_info1), transform)


    x_data_info2 = x_data[ind2]
    y_data_info2 = y_data[ind2]
    valid_tensor_dataset = CustomTensorDataset((x_data_info2,y_data_info2), transform)



    # print("··········让我康康y_trian_dict···········")
    # for i in range(len(clients_data_list)):
    #     print(i, len(clients_data_list[i]))
    #     lst = []
    #     for data, target in clients_data_list[i]:
    #         # print("target:",target)
    #         lst.append(target.item())
    #
    #     for i in range(10):  # 0-9是标签，这个需要根据不同的数据集来打印，mnist和fashionmnist是只有0-9的标签
    #         print(lst.count(i), end=' ')
    #     # print(len(client_data_dict[key].dataset.targets))
    #     print()


    return train_tensor_dataset,valid_tensor_dataset


#从训练集中划分出部分验证集,返回采样器
def dividing_validation_set2(train_data,validation_num):
    if validation_num>=len(train_data):
        raise ValueError("验证集数量超过总的训练集数量")
    indices=[x for x in range(len(train_data))]  #生成全部索引，训练集已经被打乱过的，不用再对索引打乱

    #划分训练集和验证集长度
    train_indices=indices[:len(train_data)-validation_num]
    valid_indices=indices[len(train_data)-validation_num:len(train_data)]

    #生成训练集和验证集的sampler
    train_sampler=torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler=torch.utils.data.SubsetRandomSampler(valid_indices)

    return train_sampler,valid_sampler