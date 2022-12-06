from FL_and_DP.fl_related_funtion.center_average_model_with_weights import set_averaged_weights_as_main_model_weights
from FL_and_DP.fl_related_funtion.local_clients_train_process import local_clients_train_process_without_dp_one_epoch
from FL_and_DP.fl_related_funtion.send_main_model_to_clients import send_main_model_to_clients
from data.fed_data_distribution.dirichlet_nonIID_data import fed_dataset_NonIID_Dirichlet
from FL_and_DP.fl_related_funtion.optimizier_and_model_distribution import create_model_optimizer_criterion_dict
from data.get_data import get_data
from model.CNN import CNN
from train_and_validation.validation import validation
import torch

def fed_avg(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q):

    #客户端的样本分配
    clients_data_list, weight_of_each_clients,batch_size_of_each_clients = fed_dataset_NonIID_Dirichlet(train_data,number_of_clients,alpha,seed,q)
    # # 获取各个客户端的model,optimizer,criterion的组合字典
    clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict(number_of_clients, learning_rate,
                                                                                       momentum)

    # 初始化中心模型,本质上是用来接收客户端的模型并加权平均进行更新的一个变量
    center_model = CNN()

    # #创建一个中心方有噪声和无噪声的model，optimizer和criterion
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)
    #
    print("联邦学习整体流程开始-------------------")

    for i in range(iters):

        print("现在进行和中心方的第{:3.0f}轮联邦训练".format(i+1))

        # 1 中心方广播参数给各个客户端
        model_dict = send_main_model_to_clients(center_model, clients_model_list,number_of_clients)

        # 2本地梯度下降
        local_clients_train_process_without_dp_one_epoch(number_of_clients,clients_data_list,clients_model_list,clients_criterion_list,clients_optimizer_list,numEpoch,q)

        # 3 客户端上传参数到中心方进行加权平均并更新中心方参数(根据客户端数量加权平均)
        main_model = set_averaged_weights_as_main_model_weights(center_model,clients_model_list,number_of_clients,weight_of_each_clients)

        # 查看效果中心方模型效果
        test_loss, test_accuracy = validation(main_model, test_dl)
        print("Iteration", str(i + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))


if __name__=="__main__":
    train_data, test_data = get_data('mnist', augment=False)
    model = CNN()
    #print(train_data.__dict__)
    batch_size=64
    learning_rate = 0.002
    numEpoch = 1       #客户端本地下降次数
    number_of_clients=10
    momentum=0.9
    iters=100
    alpha=0.05 #狄立克雷的异质参数
    seed=1   #狄立克雷的随机种子
    q_for_batch_size=1.0  #基于该数据采样率组建每个客户端的batchsize
    fed_avg(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q_for_batch_size)