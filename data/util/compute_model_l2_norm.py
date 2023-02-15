
#输入的是模型
def compute_model_l2norm(model):
    total_norm=0.
    params=model.state_dict()
    for key,var in params.items():
        total_norm+=(var.data.norm(2)**2).item()  #[1]表示该层的数据部分，先求l2,再把根号去掉
        #total_norm+=(torch.norm(var.clone())**2).item()  # 一样的
    total_norm=total_norm**0.5 #最后把根号加上
    return  total_norm

#输入的是字典类型
def compute_params_l2norm(params):
    total_norm=0.
    for key,var in params.items():
        total_norm+=(var.data.norm(2)**2).item()  #[1]表示该层的数据部分，先求l2,再把根号去掉
        #total_norm+=(torch.norm(var.clone())**2).item()  # 一样的
    total_norm=total_norm**0.5 #最后把根号加上
    return  total_norm
