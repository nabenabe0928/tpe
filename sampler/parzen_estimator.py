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

        weights, mus, sigmas = self._calculate(samples, lower, upper, parameters.consider_prior, parameters.prior_weight,
            parameters.consider_magic_clip, parameters.consider_endpoints, parameters.weights)
            
        self.weights = np.asarray(weights)
        self.mus = np.asarray(mus)
        self.sigmas = np.asarray(sigmas)

    def _calculate_mus(self, samples, lower_bound, upper_bound, consider_prior):
        order = np.argsort(samples)
        sorted_mus = samples[order]
        if consider_prior:
            prior_mu = 0.5 * (lower_bound + upper_bound)
            if samples.size == 0:
                sorted_mus = np.asarray([prior_mu])
                prior_pos = 0
            else:
                prior_pos = np.searchsorted(samples[order], prior_mu)
                sorted_mus = np.insert(sorted_mus, prior_pos, prior_mu)
        
        return sorted_mus, order, prior_pos

    def _calculate_sigmas(self, samples, lower_bound, upper_bound, sorted_mus, prior_pos, consider_endpoints, consider_prior, consider_magic_clip):
        if samples.size > 0:
            sorted_mus_with_bounds = np.insert([lower_bound, upper_bound], 1, sorted_mus)
            sigma = np.maximum(sorted_mus_with_bounds[1:-1] - sorted_mus_with_bounds[0:-2], sorted_mus_with_bounds[2:] - sorted_mus_with_bounds[1:-1])
            if not consider_endpoints and sorted_mus_with_bounds.size > 2:
                sigma[0] = sorted_mus_with_bounds[2] - sorted_mus_with_bounds[1]
                sigma[-1] = sorted_mus_with_bounds[-2] - sorted_mus_with_bounds[-3]
        
        maxsigma = upper_bound - lower_bound
        
        if consider_magic_clip:
            minsigma = (upper_bound - lower_bound) / min(100.0, (1.0 + len(sorted_mus)))
        else:
            minsigma = EPS
        sigma = np.clip(sigma, minsigma, maxsigma)
        
        if consider_prior:
            prior_sigma = 1.0 * (upper_bound - lower_bound)
            if samples.size > 0:
                sigma[prior_pos] = prior_sigma
            else:
                sigma = np.asarray([prior_sigma])
        
        return sigma

    def _calculate_weights(self, samples, weights_func, order, prior_pos, prior_weight, consider_prior):
        sorted_weights = weights_func(samples.size)[order]
        if consider_prior:
            sorted_weights = np.insert(sorted_weights, prior_pos, prior_weight)
        
        sorted_weights /= sorted_weights.sum()

        return sorted_weights

    def _calculate(self, samples, lower_bound, upper_bound, consider_prior, prior_weight, consider_magic_clip, consider_endpoints, weights_func):
        
        samples = np.asarray(samples)
        sorted_mus, order, prior_pos = self._calculate_mus(samples, lower_bound, upper_bound, consider_prior)
        sigma = self._calculate_sigmas(samples, lower_bound, upper_bound, sorted_mus, prior_pos, consider_endpoints, consider_prior, consider_magic_clip)
        sorted_weights = self._calculate_weights(samples, weights_func, order, prior_pos, prior_weight, consider_prior)

        return list(sorted_weights), list(sorted_mus), list(sigma)
