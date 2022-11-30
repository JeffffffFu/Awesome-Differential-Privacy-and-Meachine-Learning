
import numpy as np
import math

# 松弛差分隐私下，高级组合，和迭代次数k呈线性关系,没有采样
def compute_dp_of_advanced_composition(k,sigma,delta):

    if sigma <= 0:
        print('sigma must be larger than 0.')
    if k <= 0:
        print('k larger than 0.')
    if delta <= 0 or delta >= 1:
        print('delta must be larger than 0 and smaller than 1')
    eps = math.sqrt(16*k*np.log(1.25/delta)*np.log(1/delta)/(sigma**2))
    return eps