
import numpy as np
import math



# 
# def compute_dp_of_advanced_composition(q,k,sigma,delta):

#     if sigma <= 0:
#         print('sigma must be larger than 0.')
#     if k <= 0:
#         print('k larger than 0.')
#     if delta <= 0 or delta >= 1:
#         print('delta must be larger than 0 and smaller than 1')
#     eps = q*math.sqrt(16*k*np.log(1.25/delta)*np.log(1/delta)/(sigma**2))
#     return eps

#每一轮满足eps,delta,K轮后的组合为如下。参考《Boosting and differential privacy》
def compute_dp_of_advanced_composition(k,eps,delta,delta_new):

    if sigma <= 0:
        print('sigma must be larger than 0.')
    if k <= 0:
        print('k larger than 0.')
    if delta <= 0 or delta >= 1:
        print('delta must be larger than 0 and smaller than 1')
    eps_new =k*eps*(eps-1)+eps*math.sqrt(2*k*np.log(1/delta_new))
    delta_new = k*delta+delta_new
    return eps_new,delta_new

if __name__=="__main__":
    k=10
    eps=1
    delta=1e-5
    delta_new=1e-5
    eps_new,delta_new=compute_dp_of_advanced_composition(k,eps,delta,delta_new)
