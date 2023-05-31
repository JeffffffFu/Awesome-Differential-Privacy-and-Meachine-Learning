from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from privacy_analysis.RDP.compute_rdp import _compute_rdp
import scipy.stats
import numpy as np


def compute_multiterm_rdp(orders, num_training_steps, noise_multiplier, num_samples, max_terms_per_node, batch_size):
    terms_rv = scipy.stats.hypergeom(num_samples, max_terms_per_node, batch_size)
    terms_logprobs = [
        terms_rv.logpmf(i) for i in np.arange(max_terms_per_node + 1)
    ]

    # Compute unamplified RDPs (that is, with sampling probability = 1).
    rdp = np.array([_compute_rdp(1.0, noise_multiplier, order) for order in
                    orders])  # 暂未放大，默认使用采样率为1计算，查看dp_accounting/rdp/rdp_privacy_accountant.py:825
    unamplified_rdps = rdp

    # Compute amplified RDPs for each (order, unamplified RDP) pair.
    amplified_rdps = []
    for order, unamplified_rdp in zip(orders, unamplified_rdps):
        beta = unamplified_rdp * (order - 1)
        log_fs = beta * (
            np.square(np.arange(max_terms_per_node + 1) / max_terms_per_node))
        amplified_rdp = scipy.special.logsumexp(terms_logprobs + log_fs) / (
                order - 1)
        amplified_rdps.append(amplified_rdp)

    # Verify lower bound.
    amplified_rdps = np.asarray(amplified_rdps)
    if not np.all(unamplified_rdps *
                  (batch_size / num_samples) ** 2 <= amplified_rdps + 1e-6):
        raise ValueError('The lower bound has been violated. Something is wrong.')

    # Account for multiple training steps.
    amplified_rdps_total = amplified_rdps * num_training_steps
    unamplified_rdps_total = unamplified_rdps * num_training_steps
    return amplified_rdps_total, unamplified_rdps_total


if __name__ == "__main__":
    # orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
    orders = np.arange(1, 10, 0.1)[1:]
    num_training_steps = 10
    noise_multiplier = 1
    target_delta = 1e-5
    batch_size = 10
    num_samples = 1000
    max_terms_per_node = 1
    rdp_every_epoch, unamplified_rdps_total = compute_multiterm_rdp(orders, num_training_steps, noise_multiplier, num_samples,
                                            max_terms_per_node, batch_size)
    multiterm_epsilon, multiterm_best_alpha = compute_eps(orders, rdp_every_epoch, target_delta)
    epsilon, best_alpha = compute_eps(orders, unamplified_rdps_total, target_delta) # TODO epsilon会很大，采样率=1
    print(f"multiterm_epsilon:{multiterm_epsilon}")