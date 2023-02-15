from FL_and_DP.fl_utils.center_average_model_with_weights import set_averaged_weights_as_main_model_weights, \
    set_averaged_weights_as_main_model_weights_fully_averaged
from FL_and_DP.fl_utils.local_clients_train_process import local_clients_train_process_without_dp_one_epoch, \
    local_clients_train_process_without_dp_one_batch
from FL_and_DP.fl_utils.send_main_model_to_clients import send_main_model_to_clients
from data.fed_data_distribution.dirichlet_nonIID_data import fed_dataset_NonIID_Dirichlet
from FL_and_DP.fl_utils.optimizier_and_model_distribution import create_model_optimizer_criterion_dict
from data.fed_data_distribution.pathological_nonIID_data import pathological_split_noniid
from data.get_data import get_data
from model.CNN import CNN, Cifar10CNN
from train_and_validation.validation import validation
import torch

def fed_avg(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q,model):

    #客户端的样本分配
    clients_data_list, weight_of_each_clients,batch_size_of_each_clients =fed_dataset_NonIID_Dirichlet(train_data,number_of_clients,alpha,seed,q)
    #clients_data_list, weight_of_each_clients,batch_size_of_each_clients =pathological_split_noniid(train_data,number_of_clients,alpha,seed,q)

    # 各个客户端的model,optimizer,criterion的分配
    clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict(number_of_clients, learning_rate,model)

    # 初始化中心模型,本质上是用来接收客户端的模型并加权平均进行更新的一个变量
    center_model = model


    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    print("联邦学习整体流程开始-------------------")
    test_accuracy_record=[]
    test_loss_record=[]

    for i in range(iters):

        print("现在进行和中心方的第{:3.0f}轮联邦训练".format(i+1))

        # 1 中心方广播参数给各个客户端
        clients_model_list = send_main_model_to_clients(center_model, clients_model_list)
        # 2本地梯度下降
        local_clients_train_process_without_dp_one_batch(number_of_clients,clients_data_list,clients_model_list,clients_criterion_list,clients_optimizer_list,numEpoch,q)

        # 3 客户端上传参数到中心方进行加权平均并更新中心方参数(根据客户端数量加权平均)
        center_model = set_averaged_weights_as_main_model_weights(center_model,clients_model_list,weight_of_each_clients)

        # 查看效果中心方模型效果
        test_loss, test_accuracy = validation(center_model, test_dl)
        print("Iteration", str(i + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))

        test_loss_record.append(test_loss)
        test_accuracy_record.append(test_accuracy)

    record=[iters,numEpoch,test_loss_record,test_accuracy_record]

    torch.save(record, "../record/{}.pth".format(int(numEpoch)))

if __name__=="__main__":
    train_data, test_data = get_data('mnist', augment=False)
    #print(train_data.data)
    model = Cifar10CNN()
    model = CNN()

    #print(train_data.__dict__)
    batch_size=64
    learning_rate = 0.1
    numEpoch = 1       #客户端本地下降次数
    number_of_clients=10
    momentum=0.9
    iters=200
    alpha=0.05 #狄立克雷的异质参数
    seed=1   #随机种子
    q_for_batch_size=0.1  #基于该数据采样率组建每个客户端的batchsize
    numEpoch_list=[1,5,10,20,50]
    for numEpoch in numEpoch_list:
        iters = 200
        iters=int(iters/numEpoch)
        fed_avg(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q_for_batch_size,model)