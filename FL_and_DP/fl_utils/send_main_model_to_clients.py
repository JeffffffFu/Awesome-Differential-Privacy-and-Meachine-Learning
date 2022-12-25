import torch


# 该函数将主模型的参数发送到节点。
def send_main_model_to_clients(center_model, clients_model_list):
    with torch.no_grad():
        for i in range(len(clients_model_list)):

            clients_model_list[i].load_state_dict(center_model.state_dict(), strict=True)

    return clients_model_list