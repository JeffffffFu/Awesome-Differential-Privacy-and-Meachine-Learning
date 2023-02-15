
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#无加噪的中心化学习
# from torchvision.models import resnet18
from data.fed_data_distribution.IID_data import split_iid
from data.fed_data_distribution.pathological_nonIID_data import pathological_split_noniid
from data.get_data import get_data
from data.util.weight_initialization import init_weights
from model.CNN import CNN, Cifar10CNN, CIFAR10_CNN, CNN_tanh
from data.util.sampling import get_data_loaders_uniform_without_replace
from model.ResNet import resnet20
from train_and_validation.train import train
from train_and_validation.validation import validation

def centralization_train(train_data, test_data, batch_size, model, numEpoch, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    minibatch_size = batch_size  # 这里比较好的取值是根号n，n为每个客户端的样本数
    microbatch_size = 1  #这里默认1就好
    iterations = 1  # n个batch，这边就定一个，每次训练采样一个Lot
    minibatch_loader, microbatch_loader = get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size, iterations)     #无放回均匀采样

    #按照batch_size去分训练数据
    #这里默认作MBSGD，如果想改成BSGD，batchsize=总样本数量即可，如果想改成SGD，batchsize=1即可
    train_dl = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)


    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    print("------ Centralized Model ------")
    for epoch in range(numEpoch):
        train_dl = minibatch_loader(train_data)

        central_train_loss, central_train_accuracy = train(model, train_dl,optimizer)
        central_test_loss, central_test_accuracy = validation(model, test_dl)

        print("epoch: {:3.0f}".format(epoch + 1) + " | test loss: {:7.4f}".format(
            central_test_loss) + " | test accuracy: {:7.4f}".format(central_test_accuracy))

    print("------ Training finished ------")

if __name__=="__main__":
    train_data, test_data = get_data('mnist', augment=False)
    model = CNN_tanh()
    init_weights(model, init_type='xavier', init_gain=0.1)
    #model= resnet20(10, False)
    batch_size=256
    learning_rate = 0.002
    numEpoch = 200
    centralization_train(train_data, test_data, batch_size, model, numEpoch, learning_rate)