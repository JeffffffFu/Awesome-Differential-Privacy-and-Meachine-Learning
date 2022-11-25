from data.fed_data_distribution.dirichlet import fed_dataset_NonIID_Dirichlet
from data.get_data import get_data
from model.CNN import CNN
from train_and_validation.validation import validation


def fed_avg(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed):

    x_train_dict, y_train_dict, x_valid_dict, y_valid_dict, x_test_dict, y_test_dict, weight_of_each_clients  \
        = fed_dataset_NonIID_Dirichlet(train_data,number_of_clients,alpha,seed)

    # 获取各个客户端的model,optimizer,criterion的组合字典
    model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(number_of_clients, learning_rate,
                                                                                       momentum)

    # 获取字典联邦客户端的字典序号
    name_of_x_train_sets, name_of_y_train_sets, name_of_x_valid_sets, name_of_y_valid_sets, name_of_x_test_sets, name_of_y_test_sets, name_of_models, name_of_optimizers, name_of_criterions \
        = dict_key(x_train_dict, y_train_dict, x_valid_dict, y_valid_dict, x_test_dict, y_test_dict, model_dict,
                   optimizer_dict, criterion_dict)

    #创建一个中心方有噪声和无噪声的model，optimizer和criterion
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    print("联邦学习整体流程开始-------------------")

    for i in range(iters):

        print("现在进行和中心方的第{:3.0f}轮联邦训练".format(i+1))

        # 1 中心方广播参数给各个客户端
        model_dict = send_main_model_to_nodes_and_update_model_dict(main_model, model_dict,
                                                                    number_of_clients,name_of_models)

        # 2本地梯度下降
        nodes_local_gradient_decline(number_of_samples, 0, x_train_dict, name_of_x_train_sets \
                                                , y_train_dict, name_of_y_train_sets, batch_size, x_test_dict \
                                                , name_of_x_test_sets, y_test_dict, name_of_y_test_sets \
                                                , model_dict, name_of_models, criterion_dict, name_of_criterions \
                                                , optimizer_dict, name_of_optimizers, numEpoch)

        # 3 客户端上传参数到中心方进行加权平均并更新中心方参数(根据客户端数量加权平均)
        main_model = set_averaged_weights_as_main_model_weights_and_update_main_model_adaptive_averaged(main_model,
                                                                                                        model_dict,
                                                                                                        number_of_clients,name_of_models,weight_of_each_clients)

        # 查看效果中心方模型效果
        test_loss, test_accuracy = validation(main_model, test_dl)
        print("Iteration", str(i + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))


if __name__=="__main__":
    train_data, test_data = get_data('mnist', augment=False)
    model = CNN()
    print(train_data.__dict__)
    batch_size=64
    learning_rate = 0.002
    numEpoch = 200
    number_of_clients=10
    momentum=0.9
    iters=100
    alpha=0.05 #狄立克雷的异质参数
    seed=1   #狄立克雷的随机种子
    fed_avg(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed)