import numpy as np
from scipy import optimize
from scipy.stats import norm



#早期《Gaussian differential privacy》的方法，局限性是要求每一步的f为对称函数
#本函数与下面的函数compute_mu_possion可相互替换
#函数功能：已知训练步数steps，加噪系数sigma，采样率q ，从而计算该梯度下降法所满足的GDP的重要参数mu
def compute_mu_uniform(steps, sigma, q):

    c = q * np.sqrt(steps)
    return (
        np.sqrt(2)
        * c
        * np.sqrt(
            np.exp(sigma ** (-2)) * norm.cdf(1.5 / sigma)
            + 3 * norm.cdf(-0.5 / sigma)
            - 2
        )
    )



#compute_mu_uniform的替换版本，是《deep learning with Gaussian differential privacy》提出的改进方法
#采用泊松采样
#函数功能：已知训练步数steps，加噪系数sigma，采样率q ，从而计算该梯度下降法所满足的GDP的重要参数mu

def compute_mu_poisson(steps, sigma, q):

    return np.sqrt(np.exp(sigma ** (-2)) - 1) * np.sqrt(steps) * q




#将GDP转化为DP的重要步骤：
#该函数反映了eps，mu，delta的等式关系
#函数功能：传入eps与mu，计算delta
def delta_eps_mu(eps, mu) :

    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)



#将GDP转化为DP的重要步骤：
#该函数反映了eps，mu，delta的等式关系
#函数功能：该函数与上一步的思路类似，唯一区别是希望由传入的delta和mu来计算eps
def eps_from_mu(mu, delta):

    def f(x):
        return delta_eps_mu(eps=x, mu=mu) - delta

    return optimize.root_scalar(f, bracket=[0, 500], method="brentq").root




#实际计算eps的函数，基于的方法是compute_mu_uniform
#与下方基于compute_from_poisson的方法可互相替换
#函数的功能，传入总步数steps，加噪系数sigma，采样率q，错误率delta，返回eps
def compute_eps_uniform(steps, sigma, q, delta):

    return eps_from_mu(
        mu=compute_mu_uniform(
            steps=steps, sigma=sigma, q=q
        ),
        delta=delta,
    )



#实际计算eps的函数，基于的方法是compute_mu_poisson
#与上方基于compute_from_uniform的方法可互相替换
#函数的功能，传入总步数steps，加噪系数sigma，采样率q，错误率delta，返回eps
def compute_eps_poisson(steps, sigma, q, delta):

    return eps_from_mu(
        mu=compute_mu_poisson(
            steps=steps, sigma=sigma, q=q
        ),
        delta=delta,
    )







class GaussianAccountant():
    def __init__(self, steps, sigma, q, delta, poisson=True):
        self.steps = steps
        self.sigma = sigma
        self.q = q
        self.delta = delta
        self.poisson = poisson

    def get_epsilon(self):

        compute_eps = (
            compute_eps_poisson
            if self.poisson
            else compute_eps_uniform
        )

        return compute_eps(
            steps=self.steps,
            sigma=self.sigma,
            q=self.q,
            delta=self.delta,
        )

    def get_mu(self):

        compute_mu = (
            compute_mu_poisson
            if self.poisson
            else compute_mu_uniform
        )
        return compute_mu(
            steps=self.steps,
            sigma=self.sigma,
            q=self.q,
        )







'''
#测试 符合论文《deep learning with gaussian differential privacy》的结果
b=GaussianAccountant((60000/256)*15,1.3,256/60000,0.00001)
print(b.get_epsilon())
print(b.get_mu())
'''


