#!/user/bin/python
# author jeff
import math
# from absl import app
#整个DP-SGD的隐私分析，要从这个开始函数调用。整体来看，隐私分析和数据不需要绑定，输入对应参数即可获得，
#从这个角度看，我们不需要每次迭代都调用这个隐私函数分析

#这个DPSGD是以epochs为参数做的
#这个函数调用下面的函数，这个函数主要判断q的
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
    """Compute epsilon based on the given hyperparameters.
    Args:
      n: Number of examples in the training data. 训练集样本总数
      batch_size: Batch size used in training. 一批采样的样本数
      noise_multiplier: Noise multiplier used in training. 噪声系数
      epochs: Number of epochs in training. 本地迭代轮次（还没有算上本地一次迭代中的多个batch迭代）
      delta: Value of delta for which to compute epsilon.
      S:sensitivity      这个原本的库是没有的
    Returns:
      Value of epsilon corresponding to input hyperparameters.  返回epsilon
    """
    q = batch_size / n  # q - the sampling ratio.          这里采样率=采样样本数/总样本数
    if q > 1:
        print ('n must be larger than the batch size.')
    orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda

    # 总的steps应该是本地的epochs数*每次本地的batch迭代数（n/batch_size），因为每次batch迭代进行梯度下降
    # 由此可见，在做dataloder的时候，它也是按照这个规则去组装数据了，也就是batch_size为采样数，而每个dataloder会放n/batch_size个batch,注意batch之间应该是都要进行有放回的采样
    steps = int(math.ceil(epochs * (n / batch_size)))

    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)

#这个函数会调用计算RDP，和RDP转DP两个函数,如果只想传参steps，可以直接用这个函数而不用上面那个
def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  # compute_rdp requires that sigma be the ratio of the standard deviation of
  # the Gaussian noise to the l2-sensitivity of the function to which it is
  # added. Hence, sigma here corresponds to the `noise_multiplier` parameter   sigma=noise_multilpier
  # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
  rdp = compute_rdp(q, sigma, steps, orders)      #先算RDP、也就是RDP定义下总的隐私损失alpha，如果根据RDP算的话，可能会依据RDP的文章

  eps, opt_order = compute_eps(orders, rdp, delta)    #再根据RDP转换为对应的最佳eps和lamda

  #print('DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
  #      ' over {} steps satisfies'.format(100 * q, sigma, steps), end=' ')
  #print('differential privacy with eps = {:.3g} and delta = {}.'.format(
   #   eps, delta))
  #print('The optimal RDP order is {}.'.format(opt_order))

  # if opt_order == max(orders) or opt_order == min(orders):            #这个是一个提示可以忽略，主要告诉我们可以扩展我们的orders的范围
  #   print('The privacy estimate is likely to be improved by expanding '
  #         'the set of orders.')

  return eps, opt_order


#这个函数会调用计算RDP，和RDP转DP两个函数,如果只想传参steps，可以直接用这个函数而不用上面那个
def apply_dp_sgd_analysis_old(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  # compute_rdp requires that sigma be the ratio of the standard deviation of
  # the Gaussian noise to the l2-sensitivity of the function to which it is
  # added. Hence, sigma here corresponds to the `noise_multiplier` parameter   sigma=noise_multilpier
  # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
  rdp = compute_rdp(q, sigma, steps, orders)      #先算RDP、也就是RDP定义下总的隐私损失alpha，如果根据RDP算的话，可能会依据RDP的文章

  eps, opt_order = compute_eps(orders, rdp, delta)    #再根据RDP转换为对应的最佳eps和lamda

  #
  # if opt_order == max(orders) or opt_order == min(orders):            #这个是一个提示可以忽略，主要告诉我们可以扩展我们的orders的范围
  #   print('The privacy estimate is likely to be improved by expanding '
  #         'the set of orders.')

  return eps, opt_order

'''
orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
eps, opt_order=apply_dp_sgd_analysis(256/60000, 1.1, 17470, orders, 10**(-5))
print("eps:",format(eps)+"| order:",format(opt_order))
'''


if __name__=="__main__":
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
    eps, opt_order = apply_dp_sgd_analysis(0.05, 2.0, 1000, orders, 10 ** (-5))
    print("eps:", format(eps) + "| order:", format(opt_order))