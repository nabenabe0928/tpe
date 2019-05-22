import numpy as np
from numpy import ndarray
from typing import Callable
from typing import NamedTuple
from typing import Optional

EPS = 1e-12

class ParzenEstimatorParameters(
        NamedTuple('_ParzenEstimatorParameters', [
            ('consider_prior', bool),
            ('prior_weight', Optional[float]),
            ('consider_magic_clip', bool),
            ('consider_endpoints', bool),
            ('weights', Callable[[int], ndarray]),
        ])):
    pass

class ParzenEstimator(object):
    def __init__(self, samples, lower, upper, parameters):
        # type: (...) -> None

        weights, mus, sigmas = self._calculate(samples, lower, upper, parameters.consider_prior, parameters.prior_weight,
            parameters.consider_magic_clip, parameters.consider_endpoints, parameters.weights)
            
        self.weights = np.asarray(weights)
        self.mus = np.asarray(mus)
        self.sigmas = np.asarray(sigmas)

    def _calculate(self, samples, lower_bound, upper_bound, consider_prior, prior_weight, consider_magic_clip, consider_endpoints, weights_func):
        # type: (...) -> Tuple[List[float], List[float], List[float]]

        samples = np.asarray(samples)
        sigma = np.asarray([], dtype=float)
        prior_pos = 0
        if consider_prior:
            prior_mu = 0.5 * (lower_bound + upper_bound)
            prior_sigma = 1.0 * (upper_bound - lower_bound)
            if samples.size == 0:
                sorted_mus = np.asarray([prior_mu])
                sigma = np.asarray([prior_sigma])
                prior_pos = 0
                order = []  # type: List[int]
            else:  # When mus.size is greater than 0.
                # We decide the place of the  prior.
                order = np.argsort(samples).astype(int)
                prior_pos = np.searchsorted(samples[order], prior_mu)
                # We decide the mus.
                sorted_mus = np.zeros(len(samples) + 1)
                sorted_mus[:prior_pos] = samples[order[:prior_pos]]
                sorted_mus[prior_pos] = prior_mu
                sorted_mus[prior_pos + 1:] = samples[order[prior_pos:]]
        else:
            order = np.argsort(samples)
            # We decide the mus.
            sorted_mus = samples[order]

        # We decide the sigma.
        if samples.size > 0:
            low_sorted_mus_high = np.append(sorted_mus, upper_bound)
            low_sorted_mus_high = np.insert(low_sorted_mus_high, 0, lower_bound)
            sigma = np.zeros_like(low_sorted_mus_high)
            sigma[1:-1] = np.maximum(low_sorted_mus_high[1:-1] - low_sorted_mus_high[0:-2],
                                        low_sorted_mus_high[2:] - low_sorted_mus_high[1:-1])
            if not consider_endpoints and low_sorted_mus_high.size > 2:
                sigma[1] = low_sorted_mus_high[2] - low_sorted_mus_high[1]
                sigma[-2] = low_sorted_mus_high[-2] - low_sorted_mus_high[-3]
            sigma = sigma[1:-1]

        # We decide the weights.
        unsorted_weights = weights_func(samples.size)
        if consider_prior:
            sorted_weights = np.zeros_like(sorted_mus)
            sorted_weights[:prior_pos] = unsorted_weights[order[:prior_pos]]
            sorted_weights[prior_pos] = prior_weight
            sorted_weights[prior_pos + 1:] = unsorted_weights[order[prior_pos:]]
        else:
            sorted_weights = unsorted_weights[order]
        sorted_weights /= sorted_weights.sum()

        # We adjust the range of the 'sigma' according to the 'consider_magic_clip' flag.
        maxsigma = 1.0 * (upper_bound - lower_bound)
        if consider_magic_clip:
            minsigma = 1.0 * (upper_bound - lower_bound) / min(100.0, (1.0 + len(sorted_mus)))
        else:
            minsigma = EPS
        sigma = np.clip(sigma, minsigma, maxsigma)
        if consider_prior:
            sigma[prior_pos] = prior_sigma

        sorted_weights = list(sorted_weights)
        sorted_mus = list(sorted_mus)
        sigma = list(sigma)
        return sorted_weights, sorted_mus, sigma

