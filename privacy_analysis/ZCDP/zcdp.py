
import numpy as np
#计算高斯机制下的zcdp,无采样，采样的zcdp，需要参考文章《Differentially Private Model Publishing for Deep Learning》
def compute_zcdp(k,sigma,q):
    if sigma<=0:
        return 0
    if q==0:
        return 0
    if q==1:
        zcdp=1/(2*sigma**2)
        return k*zcdp
    else:
        print("还要再看下采样下的zcdp公式")

def zcdp_convert_dp(zcdp,delta):
    eps=zcdp + 2*np.sqrt(zcdp * np.log(1/delta))
    return eps

def compute_dp_through_zcdp(k,sigma,delta):
    zcdp=compute_zcdp(k,sigma)
    eps=zcdp_convert_dp(zcdp,delta)
    return eps