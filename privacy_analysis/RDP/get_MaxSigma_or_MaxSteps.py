

from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis


#根据目标eps，steps和采样率，反推sigma系数
def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    alphas,
    epsilon_tolerance: float = 0.01,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        steps: number of steps to run
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """

    sigma_low, sigma_high = 0, 1000  #从0-10进行搜索，一般的sigma设置也不会超过这个范围。其实从0-5就可以了我觉得。

    eps_high, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma_high, steps, alphas, target_delta)

    if eps_high>target_epsilon:
        raise ValueError("The target privacy budget is too low. 当前可供搜索的最大的sigma只到100")

    # 下面是折半搜索，直到找到满足这个eps容忍度的sigma_high,sigma是从大到小搜索，即eps从小到大逼近
    while target_epsilon - eps_high > epsilon_tolerance:   #我们希望当目前eps减去当前计算出来的eps小于容忍度，也就是计算出来的eps非常接近于目标eps
        sigma = (sigma_low + sigma_high) / 2

        eps, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps, alphas,target_delta)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return round(sigma_high,2)

#根据当前已经迭代的轮数，采样率，和之前的sigma参数，以及目标的eps，delat。计算还能迭代多少次。注意：这种计算方法要确保sigma和采样率不会变化。
def get_steps(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    sigma: float,
    alphas,
    epsilon_tolerance: float = 0.01,
) -> int:

    steps_low, steps_high = 0, 100000  #从0-100000进行搜索，一般的steps设置也不会超过这个范围。

    eps_high, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps_high, alphas, target_delta)

    if eps_high < target_epsilon:
        raise ValueError("The privacy budget is too high. 当前最大的steps搜索只到100000")

    # 下面是折半搜索，直到找到满足这个eps容忍度的steps_high,steps是从大到小搜索，即eps从大到小逼近
    while eps_high - target_epsilon > epsilon_tolerance:   #我们希望当目前eps减去当前计算出来的eps小于容忍度，也就是计算出来的eps非常接近于目标eps
        steps = (steps_low + steps_high) / 2
        eps, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps, alphas,target_delta)

        if eps > target_epsilon:
            steps_high = steps
            eps_high = eps
        else:
            steps_low = steps

    return int(steps_high)

def get_steps_without_has_runned(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    sigma: float,
    alphas,
    steps_has_runned:int,
    epsilon_tolerance: float = 0.01,
) -> int:

    eps_high, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps_has_runned, alphas, target_delta)
    if eps_high > target_epsilon:
        raise ValueError("已经跑的steps已经超过目标eps")

    steps_low, steps_high = steps_has_runned, 100000  #从steps_has_runned-100000进行搜索，一般的steps设置也不会超过这个范围。

    eps_high, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps_high, alphas, target_delta)

    if eps_high < target_epsilon:
        raise ValueError("The privacy budget is too high. 当前最大的steps搜索只到100000")

    # 下面是折半搜索，直到找到满足这个eps容忍度的steps_high,steps是从大到小搜索，即eps从大到小逼近
    while eps_high - target_epsilon > epsilon_tolerance:   #我们希望当目前eps减去当前计算出来的eps小于容忍度，也就是计算出来的eps非常接近于目标eps
        steps = (steps_low + steps_high) / 2
        eps, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps, alphas,target_delta)

        if eps > target_epsilon:
            steps_high = steps
            eps_high = eps
        else:
            steps_low = steps

    return int(steps_high)-steps_has_runned

if __name__=="__main__":
    sample_rate=512/60000
    steps=10000
    eps=3.0
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
    delta=1e-5
    sigma=1.23
    # Max_step=batch_steps_allowed(sample_rate, steps_has_runned,steps_allowed_next, maxEps, delta,sigma, alphas)
    # print("Max_step:",Max_step)
    #noise_multiplier=get_noise_multiplier(eps,delta,sample_rate,steps,alphas)
    max_steps=get_steps(eps,delta,sample_rate,sigma,alphas)

    print("max_steps:",max_steps)