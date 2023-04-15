import os

import pandas as pd

from data.fed_data_distribution.pathological_nonIID_data import pathological_split_noniid
from data.get_data import get_data
from data.util.sampling import get_data_loaders_uniform_without_replace
from model.CNN import CNN_tanh, CNN, Cifar10CNN
from model.ResNet import resnet20
from optimizer.dp_optimizer import DPSGD, DPAdam
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from train_and_validation.train import train
from train_and_validation.train_with_dp import  train_dynamic_add_noise
from train_and_validation.train_with_opacus import train_with_opacus, train_privacy_opacus2
from train_and_validation.validation import validation
from openpyxl import Workbook
import time
from opacus import PrivacyEngine




#opacus<=0.13.0
def centralization_train_with_dp_by_opacus(train_data, test_data, model,batch_size, eps_budget, learning_rate,momentum,delta,max_norm,sigma,privacy_accoutant):

    #如果用Resent模型需要加以下，将模型转换为opacus支持的
    # from opacus.utils import module_modification
    # from opacus.dp_model_inspector import DPModelInspector
    # model = module_modification.convert_batchnorm_modules(model)
    # inspector = DPModelInspector()
    # print(f"Is the model valid? {inspector.validate(model)}")


    #centralized_optimizer = torch.optim.Adam(centralized_model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)

    #按照mini_batch_size去分训练数据
    mini_batch_size=256   #并行的batch

    train_dl = torch.utils.data.DataLoader(
        train_data, batch_size=mini_batch_size, shuffle=True)

    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=mini_batch_size, shuffle=False)

    n_acc_steps=batch_size// mini_batch_size    #实际需要多少个并行的mninbatch做一次迭代

    print("------ Centralized Model ------")
    rdp=0
    # orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
    epsilon_list=[]
    result_loss_list=[]
    result_acc_list=[]

    # opacus<=0.13.0
    ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    privacy_engine = PrivacyEngine(
        model,
        sample_rate=batch_size / len(train_data),
        alphas=ORDERS,
        noise_multiplier=sigma,
        max_grad_norm=max_norm,
        accountant=privacy_accoutant
    )
    privacy_engine.attach(optimizer)
    epoch=1
    while epsilon < eps_budget:
        epoch+=1
        central_train_loss, central_train_accuracy = train_with_opacus(model, train_dl, optimizer,n_acc_steps)
        central_test_loss, central_test_accuracy = validation(model, test_dl)


        #按照一个batch进行，opacus<=0.13.0的没有prv
        epsilon = privacy_engine.accountant.get_epsilon(delta=delta)


        result_loss_list.append(central_test_loss)
        result_acc_list.append(central_test_accuracy)

        print("epoch: {:3.0f}".format(epoch + 1) + " | epsilon: {:7.4f}".format(
        epsilon))


    print("------ Training finished ------")


#opacus>=1.0.1
def centralization_train_with_dp_by_opacus2(train_data, test_data, model,batch_size, eps_budget, learning_rate,momentum,delta,max_norm,sigma,privacy_accoutant):

    from opacus.validators import ModuleValidator
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 包装抽样函数
    minibatch_size = batch_size  # 这里比较好的取值是根号n，n为每个客户端的样本数
    microbatch_size = 1  #这里默认1就好
    iterations = 1  # n个batch，这边就定一个，每次训练采样一个Lot
    minibatch_loader, microbatch_loader = get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size, iterations)     #无放回均匀采样

    # Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。
    # DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作)
    # DataLoader里的数据要通过for i ,data in enumerate(DataLoader)才能打印出来

    train_dl = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    #train数据在下面进行采样，这边不做dataloader
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    print("------ Centralized Model ------")
    epsilon_list=[]
    result_loss_list=[]
    result_acc_list=[]
    privacy_engine = PrivacyEngine(accountant=privacy_accoutant) #opacus>=1.3 提供rdp,gdp和prv三种隐私计算方式
    model_opacus, optimizer_opacus, train_dl_opacus = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_dl,
        noise_multiplier=sigma,
        max_grad_norm=max_norm,
    )
    epoch=1
    eps_97=0.
    label1=0
    eps_98=0.
    label2=0
    eps_985=0.
    label3=0
    epsilon_list=[]
    epsilon=0.
    while epsilon < eps_budget:

        epoch+=1
        central_train_loss, central_train_accuracy = train_privacy_opacus2(model_opacus, train_dl_opacus, optimizer_opacus)
        central_test_loss, central_test_accuracy = validation(model_opacus, test_dl)

        #下面的计算都是需要一个完整的epoch
        epsilon = privacy_engine.accountant.get_epsilon(delta=delta)

        if central_train_accuracy>0.97 and label1==0:
            eps_97=epsilon
            label1=1

        if central_train_accuracy>0.98 and label2==0:
            eps_98=epsilon
            label2=1

        if central_train_accuracy>0.985 and label3==0:
            eps_985=epsilon
            label3=1

        # result_loss_list.append(central_test_loss)
        # result_acc_list.append(central_test_accuracy)

        print("privacy_accoutant:",privacy_accoutant+"| epoch: {:3.0f}".format(epoch + 1) + " | epsilon: {:7.4f}".format(
        epsilon)   )

    File_Path_Csv = os.getcwd() + f"/result/csv/DPSGD/MNIST//"
    if not os.path.exists(File_Path_Csv):
        os.makedirs(File_Path_Csv)
    pd.DataFrame([eps_97, eps_98, eps_985]).to_csv(
        f"{File_Path_Csv}.csv", index=False, header=False)


    print("------ Training finished ------")

if __name__=="__main__":
    # python3
    # cnns.py - -dataset = mnist - -batch_size = 512 - -lr = 0.5 - -noise_multiplier = 1.23
    # python3
    # cnns.py - -dataset = fmnist - -batch_size = 2048 - -lr = 4 - -noise_multiplier = 2.15
    # python3
    # cnns.py - -dataset = cifar10 - -batch_size = 1024 - -lr = 1 - -noise_multiplier = 1.54

    #train_data, test_data = get_data('cifar10', augment=False)
    train_data, test_data = get_data('mnist', augment=False)
   #  print(train_data.__dict__)
    model = CNN()
    #model= resnet20(10, False)  #含有batchNorm层的需要进行模型转换
    batch_size =512
    learning_rate = 0.5
    eps_budget = 5.0
    sigma = 1.23
    momentum = 0.9
    delta = 10 ** (-5)
    max_norm =0.1
    privacy_accoutant='rdp' #rdp,gdp,prv(opacus>=1.3.0)
    centralization_train_with_dp_by_opacus2(train_data, test_data, model,batch_size, eps_budget, learning_rate,momentum,delta,max_norm,sigma,privacy_accoutant)