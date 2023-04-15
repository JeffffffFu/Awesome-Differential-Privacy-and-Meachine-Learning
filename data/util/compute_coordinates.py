import numpy as np
import torch
import math
import numpy as np

from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.RDP.get_MaxSigma_or_MaxSteps import get_noise_multiplier


def cartesian_to_polar(x):
    r = np.linalg.norm(x)
    theta = np.arccos(x[0] / r)  #值域范围是[0,pi]
    phi = [1. for i in range(len(x) - 1)]
    for i in range(len(phi)):
        phi[i] = np.arctan2(x[i + 1], x[0])  # 以X[0]为标准计算反正切值,#值域范围是[-pi,pi]
    return np.concatenate(([r, theta], phi))


def polar_to_cartesian(p):
    r = p[0]
    theta = p[1]
    phi = p[2:]
    x = [1. for i in range(len(phi) + 1)]
    x[0] = r * np.cos(theta)  # 用这个求回X[0]没有问题
    for i in range(len(phi)):
        x[i + 1] = x[0] * np.tan(phi[i])
    for j in range(len(x)):
        x[j]=round(x[j],4)  #保留小数点后四位
    return x

#划分eps
def devide_epslion(sigma,q,n):
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    eps, opt_order = apply_dp_sgd_analysis(q, sigma, 1, orders, 10 ** (-5))
    #print("初始状态下每个梯度元素的eps:",eps)
    eps_sum = n * eps
   # print("LDP定义下的eps_sum:", eps_sum)
    eps1=eps_sum*0.000001
   # print("分给极值的eps1:",eps1)
    eps2=eps_sum-eps1
  #  print("分给每一个角度的eps2:",eps2)
    sigma1=get_noise_multiplier(target_epsilon=eps1,target_delta=1e-5,sample_rate=512 / 60000,steps=1,alphas=orders)
    sigma2=get_noise_multiplier(target_epsilon=eps2,target_delta=1e-5,sample_rate=512 / 60000,steps=1,alphas=orders)
    return sigma1,sigma2

#加噪
def cartesian_add_noise(p,sigma1,C1,sigma2):
    #print("sigma1:{}".format(sigma1)+"| sigma2:{}".format(sigma2))
    r = p[0]

    #对极值加噪，因为对梯度裁剪的时候其实就已经裁剪了
    r+=C1 * sigma1 * np.random.normal(0, 1)


    theta = p[1:]  #默认在-pi到pi之间，也就是2*pi
    theta+=2*math.pi*sigma2* np.random.normal(0, 1)

    return np.concatenate(([r], theta))

def vector_to_matrix(vector, shape):
    shape=tuple(shape)
    if len(shape) == 0 or np.prod(shape) != len(vector):
        raise ValueError("Invalid input dimensions")
    matrix = np.zeros(shape)
    strides = [np.prod(shape[i+1:]) for i in range(len(shape)-1)] + [1]
    for i in range(len(vector)):
        index = [0] * len(shape)
        for j in range(len(shape)):
            index[j] = (i // strides[j]) % shape[j]
        matrix[tuple(index)] = vector[i]
    return matrix