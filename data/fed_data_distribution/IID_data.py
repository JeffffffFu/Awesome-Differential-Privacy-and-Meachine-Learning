import torch
import torchvision

from data.util.custom_tensor_dataset import CustomTensorDataset

# https://github.com/vaseline555/Federated-Averaging-PyTorch/blob/main/src/utils.py
def split_iid(train_data, n_clients):

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


    # shuffle data
    shuffled_indices = torch.randperm(len(train_data))
    training_inputs = train_data.data[shuffled_indices]
    training_labels = torch.Tensor(train_data.targets)[shuffled_indices]

    # partition data into num_clients
    split_size = len(train_data) // n_clients
    split_datasets = list(
        zip(
            torch.split(torch.Tensor(training_inputs), split_size),  #split函数自动把总样本和标签切成n客户端组，每组split_size个
            torch.split(torch.Tensor(training_labels), split_size)
        )
    )

    # finalize bunches of local datasets
    clients_data_list = [
        CustomTensorDataset(local_dataset, transform=transform)
        for local_dataset in split_datasets
    ]

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

    return clients_data_list