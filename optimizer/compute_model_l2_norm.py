import torch


def compute_model_l2norm(model):
    total_norm=0.
    params=model.state_dict()
    for key,var in params.items():
        value=torch.tensor(var.data.clone().detach(),dtype = torch.double)
        total_norm+=value.norm(2)**2  #[1]表示该层的数据部分，先求l2,再把根号去掉
    total_norm=total_norm**0.5 #最后把根号加上
    return  total_norm