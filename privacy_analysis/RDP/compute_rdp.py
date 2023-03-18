#!/user/bin/python
# author jeff
import numpy as np
import math
from scipy import special

from privacy_analysis.RDP.rdp_convert_dp import compute_eps


def compute_rdp(q, noise_multiplier, steps, orders):
  """Computes RDP of the Sampled Gaussian Mechanism.
  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise    STD标准差，敏感度应该包含在这里面了
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  if np.isscalar(orders):        #判断orders是不是标量类型，判断是否为一个数字还是一组list，只有一个数字走这个
    rdp = _compute_rdp(q, noise_multiplier, orders)     #这里是具体计算采样下的RDP隐私损失的
  else:                       #如果是一个list，走的是这个函数，一般走这个
    rdp = np.array(
        [ _compute_rdp(q, noise_multiplier, order) for order in orders])

  return rdp * steps     #这里直接乘以总的迭代次数即可

#整型
def _compute_log_a_for_int_alpha(q, sigma, alpha):
    assert isinstance(alpha, int)
    rdp = -np.inf

    for i in range(alpha + 1):
        log_b = (
                math.log(special.binom(alpha, i))
                + i * math.log(q)
                + (alpha - i) * math.log(1 - q)
                + (i * i - i) / (2 * (sigma ** 2))
        )

        # rdp=math.exp(log_b)+math.exp(rdp)           # 当加到后面，math.exp计算的数字以小数表示，超过110000位数。超出了Double的范围，会导致溢出。所以我们用下面的方法

        # 这边其实和上面我注释的等价，这里做了一些数值超出范围的处理
        a, b = min(rdp, log_b), max(rdp, log_b)
        if a == -np.inf:  # adding 0
            rdp = b
        else:
            rdp = math.log(math.exp(
                a - b) + 1) + b  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b),这里为什么不直接exp(a) + exp(b) ，可能是容易超出数值？

    rdp = float(rdp) / (alpha - 1)
    return rdp


def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.

    Args:
        logx: First term in log space.
        logy: Second term in log space.

    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.

    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.

    Returns:
        Difference of numbers in log space.

    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx

def _log_erfc(x: float) -> float:
    r"""Computes :math:`log(erfc(x))` with high accuracy for large ``x``.

    Helper function used in computation of :math:`log(A_\alpha)`
    for a fractional alpha.

    Args:
        x: The input to the function

    Returns:
        :math:`log(erfc(x))`
    """
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)

def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1) / (alpha - 1)


# 这里计算RDP，是没有敏感度这个参数的，本质上是有敏感度参数，分子和分母会恰好把敏感度消掉，也就是这里的参数sigma是不包括敏感度的
# 具体可以看这个的讨论：https://discuss.pytorch.org/t/how-to-adjusting-the-noise-increase-parameter-for-each-round/143548/17
def _compute_rdp(q, sigma, alpha):
    """Compute RDP of the Sampled Gaussian mechanism at order alpha.
    Args:
      q: The sampling rate.
      sigma: The std of the additive Gaussian noise.
      alpha: The order at which RDP is computed.
    Returns:
      RDP at alpha, can be np.inf.

      q==1时的公式可参考：[renyi differential privacy,2017,Proposition 7]
      0<q<1时，有以下两个公式：
      可以参考[Renyi Differential Privacy of the Sampled Gaussian Mechanism ,2019,3.3]，这篇文章中包括alpha为浮点数的计算
      公式2更为简洁的表达在[User-Level Privacy-Preserving Federated Learning: Analysis and Performance Optimization,2021,3.2和3.3]
    """
    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    #q=1时相当于没有抽样，这里大家会质疑为什么没有处以（alpha-1）,其实该系数已经被消掉了(see proposition 7 of https://arxiv.org/pdf/1702.07476.pdf 1)
    if q == 1.:  # 相当于没有抽样
        return alpha  / (2 * sigma ** 2)  # 没有抽样下参照RDP两个高斯的瑞丽分布，应该是 alpha*s  / (2 * sigma**2),这边为什么少了一个敏感度S，是默认函数敏感度为1吗，答案是敏感度和抵消了，这边的sigma里面没有敏感度的

    if np.isinf(alpha):
        return np.inf

    if float(alpha).is_integer(): #整型
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:  #浮点型
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)

#[renyi differential privacy,2017,Proposition 5] 随机扰动没有抽样的
#这里的入参就算扰动概率，和一组alpha，steps默认为1
def compute_rdp_randomized_response(p,steps,orders,q):

    if np.isscalar(orders):  # 判断orders是不是标量类型，判断是否为一个数字还是一组list，只有一个数字走这个
        rdp = _compute_rdp_randomized_response(p, orders)  # 这里是具体计算采样下的RDP隐私损失的
    else:  # 如果是一个list，走的是这个函数，一般走这个
        rdp = np.array([_compute_rdp_randomized_response(p, order) for order in orders])

    return q*rdp * steps  # 这里直接乘以总的迭代次数即可

def _compute_rdp_randomized_response(p,alpha):
    item1=float((p**alpha)*((1-p)**(1-alpha)))
   # print("item1:",item1)
    item2=float(((1-p)**alpha)*(p**(1-alpha)))
   # print("item2:",item2)
  #  print("a:",alpha)
   # print("item1+item2:",item1+item2)
    rdp=float(math.log(item1+item2))/ (alpha-1)
    return rdp


if __name__ == '__main__':
    ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 128))
    p=0.7
    steps=100
    q=3000/60000
    delta=1e-5
    rdp=compute_rdp_randomized_response(p, steps, ORDERS, q)
    dp,ord=compute_eps(ORDERS, rdp, delta)

    print("rdp:",dp)
    print("ORDERS:",ord)
