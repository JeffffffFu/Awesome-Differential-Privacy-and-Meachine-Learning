import math

from FL_and_DP.fl_related_funtion.center_average_model_with_weights import set_averaged_weights_as_main_model_weights, \
    set_averaged_weights_as_main_model_weights_fully_averaged
from FL_and_DP.fl_related_funtion.local_clients_train_process import local_clients_train_process_without_dp, \
    local_clients_train_process_with_dp
from FL_and_DP.fl_related_funtion.send_main_model_to_clients import send_main_model_to_clients
from data.fed_data_distribution.dirichlet_nonIID_data import fed_dataset_NonIID_Dirichlet
from FL_and_DP.fl_related_funtion.optimizier_and_model_distribution import create_model_optimizer_criterion_dict, \
    create_model_optimizer_criterion_dict_with_dp_optimizer
from data.get_data import get_data
from model.CNN import CNN
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from train_and_validation.validation import validation
import torch

#样本级别的差分隐私联邦学习，在每个客户端本地梯度下降的过程中加入CDP,这个是严格按照epoch以及batch分割的定义进行的
#在很多文章中batch的随机排列组合和不放回抽样在RDP的结论效果是一样的
def fed_avg_with_dp_sample_level(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q,max_norm,sigma,delta):

    #客户端的样本分配
    clients_data_list, weight_of_each_clients,batch_size_of_each_clients = fed_dataset_NonIID_Dirichlet(train_data,number_of_clients,alpha,seed,q)


    # # 获取各个客户端的model,optimizer,criterion的组合字典
    clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict_with_dp_optimizer(number_of_clients, learning_rate,
                                                                                       momentum,max_norm,sigma,batch_size_of_each_clients)


    # 初始化中心模型,本质上是用来接收客户端的模型并加权平均进行更新的一个变量
    center_model = CNN()

    # #创建一个中心方有噪声和无噪声的model，optimizer和criterion
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)


    orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
    rdp=0.
    print("联邦学习整体流程开始-------------------")

    for i in range(iters):

        print("现在进行和中心方的第{:3.0f}轮联邦训练".format(i+1))

        # 1 中心方广播参数给各个客户端
        model_dict = send_main_model_to_clients(center_model, clients_model_list)

        # 2本地梯度下降需要加噪
        local_clients_train_process_with_dp(number_of_clients,clients_data_list,clients_model_list,clients_criterion_list,clients_optimizer_list,numEpoch,q)

        #这边计算rdp的次数，一个是看多少个epoch,一个是看一个epoch里面做了多少次batch的迭代，这边应该是1/q向下取整，因为我们在组装datasize的时候会trop_out
        rdp += compute_rdp(q, sigma, numEpoch*math.floor(1/q), orders)
        epsilon, best_alpha = compute_eps(orders, rdp, delta)
        print("Iteration: {:3.0f}".format(i + 1) + " | epsilon: {:7.4f}".format(
            epsilon) + " | best_alpha: {:7.4f}".format(best_alpha))

        # 3 客户端上传参数到中心方进行加权平均并更新中心方参数(根据客户端数量加权平均)
       # main_model = set_averaged_weights_as_main_model_weights(center_model,clients_model_list,number_of_clients,weight_of_each_clients)

        main_model = set_averaged_weights_as_main_model_weights_fully_averaged(center_model,clients_model_list,weight_of_each_clients)


        # 查看效果中心方模型效果
        test_loss, test_accuracy = validation(main_model, test_dl)
        print("Iteration", str(i + 1), ": main_model accuracy on all test data: {.2f}%".format(test_accuracy))


def fed_avg_with_dp_sample_level_many_batch(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q,max_norm,sigma,delta):

    #客户端的样本分配
    clients_data_list, weight_of_each_clients,batch_size_of_each_clients = fed_dataset_NonIID_Dirichlet(train_data,number_of_clients,alpha,seed,q)

    # # 获取各个客户端的model,optimizer,criterion的组合字典
    clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict_with_dp_optimizer(number_of_clients, learning_rate,
                                                                                       momentum,max_norm,sigma,batch_size_of_each_clients)


    # 初始化中心模型,本质上是用来接收客户端的模型并加权平均进行更新的一个变量
    center_model = CNN()

    # #创建一个中心方有噪声和无噪声的model，optimizer和criterion
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)


    orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
    rdp=0.
    print("联邦学习整体流程开始-------------------")

    for i in range(iters):

        print("现在进行和中心方的第{:3.0f}轮联邦训练".format(i+1))

        # 1 中心方广播参数给各个客户端
        model_dict = send_main_model_to_clients(center_model, clients_model_list,number_of_clients)

        # 2本地梯度下降需要加噪
        local_clients_train_process_with_dp(number_of_clients,clients_data_list,clients_model_list,clients_criterion_list,clients_optimizer_list,numEpoch,q)

        #这边计算rdp的次数，一个是看多少个epoch,一个是看一个epoch里面做了多少次batch的迭代，这边应该是1/q向下取整，因为我们在组装datasize的时候会trop_out
        rdp += compute_rdp(q, sigma, numEpoch*math.floor(1/q), orders)
        epsilon, best_alpha = compute_eps(orders, rdp, delta)
        print("Iteration: {:3.0f}".format(i + 1) + " | epsilon: {:7.4f}".format(
            epsilon) + " | best_alpha: {:7.4f}".format(best_alpha))

        # 3 客户端上传参数到中心方进行加权平均并更新中心方参数(根据客户端数量加权平均)
        main_model = set_averaged_weights_as_main_model_weights(center_model,clients_model_list,number_of_clients,weight_of_each_clients)

        # 查看效果中心方模型效果
        test_loss, test_accuracy = validation(main_model, test_dl)
        print("Iteration", str(i + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))

if __name__=="__main__":
    train_data, test_data = get_data('mnist', augment=False)
    model = CNN()
    batch_size=64
    learning_rate = 0.002
    numEpoch = 1       #客户端本地下降次数
    number_of_clients=10
    momentum=0.9
    iters=100
    alpha=0.05 #狄立克雷的异质参数
    seed=1   #狄立克雷的随机种子
    q_for_batch_size=0.01   #基于该数据采样率组建每个客户端的batchsize
    max_norm=0.1
    sigma=1.1
    delta=1e-5
    fed_avg_with_dp_sample_level(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q_for_batch_size,max_norm,sigma,delta)