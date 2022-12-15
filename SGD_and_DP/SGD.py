
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#无加噪的中心化学习
from data.fed_data_distribution.pathological_nonIID_data import pathological_split_noniid
from data.get_data import get_data
from model.CNN import CNN, Cifar10CNN, CIFAR10_CNN
from data.util.sampling import get_data_loaders_uniform_without_replace
from train_and_validation.train import train
from train_and_validation.validation import validation

def centralization_train(train_data, test_data, batch_size, model, numEpoch, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # print("train_data:",train_data[1])

    # Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。
    # DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作)
    # DataLoader里的数据要通过for i ,data in enumerate(DataLoader)才能打印出来

    #按照batch_size去分训练数据
    #这里默认作MBSGD，如果想改成BSGD，batchsize=总样本数量即可，如果想改成SGD，batchsize=1即可
    train_dl = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    print("------ Centralized Model ------")
    for epoch in range(numEpoch):

        central_train_loss, central_train_accuracy = train(model, train_dl, criterion,optimizer)
        # central_test_loss, central_test_accuracy = validation(model, test_dl)
        central_test_loss, central_test_accuracy = validation(model, test_dl)


        print("epoch: {:3.0f}".format(epoch + 1) + " | test loss: {:7.4f}".format(
            central_test_loss) + " | test accuracy: {:7.4f}".format(central_test_accuracy))

    print("------ Training finished ------")

if __name__=="__main__":
    train_data, test_data = get_data('mnist', augment=False)
    model = CNN()
    batch_size=64
    learning_rate = 0.002
    numEpoch = 200
    centralization_train(train_data, test_data, batch_size, model, numEpoch, learning_rate)