
import numpy as np
import math

# 松弛差分隐私下，高级组合，和迭代次数k及q呈线性关系
def compute_dp_of_advanced_composition(q,k,sigma,delta):

    if sigma <= 0:
        print('sigma must be larger than 0.')
    if k <= 0:
        print('k larger than 0.')
    if delta <= 0 or delta >= 1:
        print('delta must be larger than 0 and smaller than 1')
    eps = q*math.sqrt(16*k*np.log(1.25/delta)*np.log(1/delta)/(sigma**2))
    return eps

if __name__=="__main__":
    q=64/20000
    k=31300
    sigma=3.29
    delta=1e-5
    eps=compute_dp_of_advanced_composition(q, k, sigma, delta)
    print("eps:",eps)