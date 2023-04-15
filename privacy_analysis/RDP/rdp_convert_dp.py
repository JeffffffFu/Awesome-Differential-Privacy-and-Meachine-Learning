#!/user/bin/python
# author jeff
#由给定的delta去算eps,相反的我们暂时不做
#[Mironov, 2017, Propisition 3]是RDP转DP原始的公式
#[Hypothesis Testing Interpretations and Rényi Differential Privacy,2020, Theorem 21 ]给了给紧凑的RDP转DP的公式如下,这个推断大概看一下


import math
import numpy as np
def compute_eps(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  orders_vec = np.atleast_1d(orders) #输入转换为至少一维的数组
  rdp_vec = np.atleast_1d(rdp)

  if delta <= 0:        #delat不能小于等于0
    raise ValueError("Privacy failure probability bound delta must be >0.")
  if len(orders_vec) != len(rdp_vec):     #两个数组的长度需要相等
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  # #[Mironov, 2017, Propisition 3]是RDP转DP原始的公式，可能不够紧凑
  #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

  #[Hypothesis Testing Interpretations and Rényi Differential Privacy,2020, Theorem 21 ]给了给紧凑的RDP转DP的公式如下
  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
  # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
  eps_vec = []
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1:
      raise ValueError("Renyi divergence order must be >=1.")
    if r < 0:
      raise ValueError("Renyi divergence must be >=0.")

    if delta**2 + math.expm1(-r) >= 0:        #delta的约束条件
      # In this case, we can simply bound via KL divergence:
      # delta <= sqrt(1-exp(-KL)).
      eps = 0  # No need to try further computation if we have eps = 0.
    elif a > 1.01:
      # This bound is not numerically stable as alpha->1.Thus we have a min value of alpha.
      eps = ( r - (np.log(delta) + np.log(a)) / (a - 1) + np.log((a - 1) / a))
    else:
      # In this case we can't do anything. E.g., asking for delta = 0.
      eps = np.inf     #无穷大
    eps_vec.append(eps)


  idx_opt = np.argmin(eps_vec)   #找一个最小的
  return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


# 下面这个是最初的RDP转DP的公式，2020年之前很多文章，包括opacus的老版本应该用的是这个
# Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
# #[Mironov, 2017, Propisition 3]是RDP转DP原始的公式，可能不够紧凑
#   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )
def compute_eps2(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
  multiple RDP orders and target ``delta``.

  Args:
      orders: An array (or a scalar) of orders (alphas).
      rdp: A list (or a scalar) of RDP guarantees.
      delta: The target delta.

  Returns:
      Pair of epsilon and optimal order alpha.

  Raises:
      ValueError
          If the lengths of ``orders`` and ``rdp`` are not equal.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if len(orders_vec) != len(rdp_vec):
    raise ValueError(
      f"Input lists must have the same length.\n"
      f"\torders_vec = {orders_vec}\n"
      f"\trdp_vec = {rdp_vec}\n"
    )

  eps = rdp_vec - math.log(delta) / (orders_vec - 1)

  # special case when there is no privacy
  if np.isnan(eps).all():
    return np.inf, np.nan

  idx_opt = np.nanargmin(eps)  # Ignore NaNs
  return eps[idx_opt], orders_vec[idx_opt]