
import numpy as np
#计算高斯机制下的zcdp,无采样，采样的zcdp，参考文章《Differentially Private Model Publishing for Deep Learning》

#《Differentially Private Model Publishing for Deep Learning》说到对于随机排列，也就是一个epoch里面分batch，文章说其实就说采样不放回，这种在zcdp中直接等于每个batch的max,文章将其看成并行，没用做隐私预算的类和
# 这里的epoch就可以看成MBSGD中做完全部样本的一个epoch
def compute_zcdp_random_reshuffling(epoch,sigma):
    if sigma<=0:
        return 0
    zcdp=1/(2*sigma**2)
    return epoch*zcdp

#《Differentially Private Model Publishing for Deep Learning》说到对于随机采样放回，没有隐私放大的效果，所有多少的采样率都一样
#这里的Iters，我们认为是所有batch的训练次数，和epoch不一样
def compute_zcdp_sampling_with_replacement(iters,sigma):
    if sigma<=0:
        return 0
    zcdp=1/(2*sigma**2)
    return iters*zcdp

def zcdp_convert_dp(zcdp,delta):
    eps=zcdp + 2*np.sqrt(zcdp * np.log(1/delta))
    return eps

def compute_dp_through_zcdp_random_reshuffling(k,sigma,delta):
    zcdp=compute_zcdp_random_reshuffling(k,sigma)
    eps=zcdp_convert_dp(zcdp,delta)
    return eps

def compute_dp_through_zcdp_sampling_with_replacement(k,sigma,delta):
    zcdp=compute_zcdp_random_reshuffling(k,sigma)
    eps=zcdp_convert_dp(zcdp,delta)
    return eps