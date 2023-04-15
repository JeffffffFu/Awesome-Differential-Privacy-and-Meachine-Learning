import copy
import os

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#无加噪的中心化学习
# from torchvision.models import resnet18
from data.fed_data_distribution.IID_data import split_iid
from data.fed_data_distribution.pathological_nonIID_data import pathological_split_noniid, dividing_validation_set
from data.get_data import get_data
from data.util.weight_initialization import init_weights
from model.CNN import CNN, Cifar10CNN, CIFAR10_CNN, CNN_tanh
from data.util.sampling import get_data_loaders_uniform_without_replace, get_data_loaders_possion
from model.DNN import DNN
from model.ResNet import resnet20
from model.vgg_bn import vgg19_bn
from train_and_validation.train import train
from train_and_validation.validation import validation

def centralization_train(train_dataset, test_data, batch_size, model, numEpoch, learning_rate):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True,
    #                                                  threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0,
    #                                                  eps=1e-08)


    minibatch_loader, microbatch_loader = get_data_loaders_possion(batch_size, 1, 1)     #无放回均匀采样

    train_data, valid_data = dividing_validation_set(train_dataset, 5000)

    #按照batch_size去分训练数据
    #这里默认作MBSGD，如果想改成BSGD，batchsize=总样本数量即可，如果想改成SGD，batchsize=1即可
    # train_dl = torch.utils.data.DataLoader(
    #     train_data, batch_size=batch_size, shuffle=True)


    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)
    valid_dl = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False)

    last_central_valid_loss = 100000.0
    t=0
    last_centralized_model = model

    list_t=[]
    list_epoch=[]
    test_loss_list=[]
    test_acc_list=[]
    result_list=[]
    epoch=1

    print("------ Centralized Model ------")
    for t in range(numEpoch):
        train_dl = minibatch_loader(train_data)
        #valid_dl = minibatch_loader(valid_data)

        central_train_loss, central_train_accuracy = train(model, train_dl,optimizer)
        central_valid_loss, central_valid_accuracy = validation(model, valid_dl)
        central_test_loss, central_test_accuracy = validation(model, test_dl)

        print("epoch: {:3.0f}".format(epoch + 1) + " | test loss: {:7.4f}".format(
            central_test_loss) + " | test accuracy: {:7.4f}".format(central_test_accuracy))

        deltaE = central_valid_loss - last_central_valid_loss
        print("损失值之差：",deltaE)

        if deltaE<0 :
            print("接受模型更新")
            t+=1
            last_central_valid_loss = central_valid_loss
            last_centralized_model = copy.deepcopy(model)
            test_loss_list.append(central_test_loss)
            test_acc_list.append(central_test_accuracy)
            list_t.append(t)
        else:
            print("不接受模型更新")
            model.load_state_dict(last_centralized_model.state_dict(), strict=True)

        epoch+=1
        list_epoch.append(epoch)

    result_list.append(list_t)
    result_list.append(test_loss_list)
    result_list.append(test_acc_list)
    result_list.append(list_epoch)

    File_Path_Csv = os.getcwd() + f"/result//"
    torch.save(result_list,f"{File_Path_Csv}/result.pt")

    print("------ Training finished ------")

if __name__=="__main__":
    train_data, test_data = get_data('mnist', augment=False)
    model = CNN()
    #model = vgg19_bn(input_channel=1, num_classes=10)
    #init_weights(model, init_type='xavier', init_gain=0.1)
    #model= resnet20(10, False)
    batch_size=256
    learning_rate = 0.01
    numEpoch = 1000
    centralization_train(train_data, test_data, batch_size, model, numEpoch, learning_rate)