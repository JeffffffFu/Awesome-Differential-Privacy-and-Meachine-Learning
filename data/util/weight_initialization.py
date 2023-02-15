import torch.nn.init as init
import torch
import torch.nn as nn

from data.util.compute_model_l2_norm import compute_model_l2norm
from model.CNN import CNN_tanh


def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.

    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).越小，得到的权值范数越小。

    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.

    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)

    Returns:
        An initialized torch.nn.Module instance.
    """

    #gpu_ids目前我们没用到
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model

if __name__=="__main__":
    model=CNN_tanh()
    init_weights(model,init_type='xavier',init_gain=0.1)
    model_norm=compute_model_l2norm(model)
    print("model_norm:",model_norm)