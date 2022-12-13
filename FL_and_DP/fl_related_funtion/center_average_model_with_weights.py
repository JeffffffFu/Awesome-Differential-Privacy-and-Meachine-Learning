import torch


def set_averaged_weights_as_main_model_weights(center_model,clients_model_list,number_of_clients,weight_of_each_clients):
    sum_parameters = None  # 用来接所有边缘节点的模型的参数
    global_parameters = {}
    for key, var in center_model.state_dict().items():
        global_parameters[key] = var.clone()

    with torch.no_grad():

        for i in range(number_of_clients):

            local_parameters = clients_model_list[i].state_dict()  # 先把第i个客户端的model取出来

            if sum_parameters is None:  # 先初始化模型字典，主要是初始化key
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = weight_of_each_clients[i] * var.clone()

            else:  # 然后做值的累加,这边直接加权了
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + weight_of_each_clients[i] * local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var])

    center_model.load_state_dict(global_parameters, strict=True)
    return center_model

#简单的平均，不做加权
def set_averaged_weights_as_main_model_weights_fully_averaged(center_model,clients_model_list,number_of_clients,weight_of_each_clients):
    sum_parameters = None  # 用来接所有边缘节点的模型的参数
    global_parameters = {}
    for key, var in center_model.state_dict().items():
        global_parameters[key] = var.clone()

    with torch.no_grad():

        for i in range(number_of_clients):

            local_parameters = clients_model_list[i].state_dict()  # 先把第i个客户端的model取出来

            if sum_parameters is None:  # 先初始化模型字典，主要是初始化key
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()

            else:  # 然后做值的累加,这边直接加权了
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] +  local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / len(clients_model_list))

    center_model.load_state_dict(global_parameters, strict=True)
    return center_model