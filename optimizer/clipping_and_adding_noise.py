import torch


def clipping_and_adding_noise(model,max_norm,noise_scale):
    model_parameter_grad_norm=0.
    per_data_parameters_grad_dict={}
    params=model.state_dict()
    with torch.no_grad():
        #计算范数
        for key,var in params.items():
            model_parameter_grad_norm+=var.data.norm(2)**2
            per_data_parameters_grad_dict[key]=var.clone().detach()
        model_parameter_grad_norm=model_parameter_grad_norm ** 0.5

        #print("model_parameter_grad_norm:",model_parameter_grad_norm)
        #逐层裁剪并加噪
        for key in per_data_parameters_grad_dict:
            per_data_parameters_grad_dict[key] /= max(1,model_parameter_grad_norm/max_norm)
            per_data_parameters_grad_dict[key] += max_norm * noise_scale * torch.randn_like(per_data_parameters_grad_dict[key])


        #问题出现在这个model.load_state_dict,我们看一下具体是什么问题
        model.load_state_dict(per_data_parameters_grad_dict, strict=True)
    return model