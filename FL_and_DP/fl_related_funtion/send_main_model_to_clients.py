import torch


# 该函数将主模型的参数发送到节点。
def send_main_model_to_clients(center_model, clients_model_list,number_of_clients):
    with torch.no_grad():
        for i in range(number_of_clients):

            clients_model_list[i].load_state_dict(center_model.state_dict(), strict=True)

    return clients_model_list