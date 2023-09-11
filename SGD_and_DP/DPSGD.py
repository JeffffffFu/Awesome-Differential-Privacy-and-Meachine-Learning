from data.fed_data_distribution.pathological_nonIID_data import pathological_split_noniid
from data.get_data import get_data
from data.util.compute_model_l2_norm import compute_model_l2norm
from data.util.sampling import get_data_loaders_uniform_without_replace
from data.util.weight_initialization import init_weights
from model.CNN import CNN_tanh, CNN, CIFAR10_CNN
from model.ResNet import resnet20
from optimizer.dp_optimizer import DPSGD, DPAdam
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from train_and_validation.train_with_dp import  train_dynamic_add_noise
from train_and_validation.validation import validation
from openpyxl import Workbook
import time

def centralization_train_with_dp(train_data, test_data, model,batch_size, numEpoch, learning_rate,momentum,delta,max_norm,sigma):


    optimizer = DPSGD(
        l2_norm_clip=max_norm,      #裁剪范数
        noise_multiplier=sigma,
        minibatch_size=batch_size,        #几个样本梯度进行一次梯度下降
        microbatch_size=1,                #几个样本梯度进行一次裁剪，这里选择逐样本裁剪
        params=model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    # 包装抽样函数
    minibatch_size = batch_size  # 这里比较好的取值是根号n，n为每个客户端的样本数
    microbatch_size = 1  #这里默认1就好
    iterations = 1  # n个batch，这边就定一个，每次训练采样一个Lot
    minibatch_loader, microbatch_loader = get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size, iterations)     #无放回均匀采样


    #train数据在下面进行采样，这边不做dataloader
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    print("------ Centralized Model ------")
    rdp=0
    orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
    epsilon_list=[]
    result_loss_list=[]
    result_acc_list=[]
    for epoch in range(numEpoch):

        train_dl = minibatch_loader(train_data)     #抽样

    #这里要动态加噪，每次传入的sigma可能会改变
        central_train_loss, central_train_accuracy = train_dynamic_add_noise(model, train_dl, optimizer)
        central_test_loss, central_test_accuracy = validation(model, test_dl)


        if (epoch > 6200):
            break


    print("------ Training finished ------")

if __name__=="__main__":

    train_data, test_data = get_data('mnist', augment=False)
    model = CNN_tanh()
    batch_size = 512
    learning_rate =0.5
    numEpoch = 15000
    sigma = 1.23
    momentum = 0.9
    delta = 10 ** (-5)
    max_norm =0.1
    centralization_train_with_dp(train_data, test_data, model,batch_size, numEpoch, learning_rate,momentum,delta,max_norm,sigma)