import numpy as np
from scipy.special import logsumexp


def compute_config_loglikelihoods(basis_loglikelihoods: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Calculate the loglikelihood of configurations based on parzen estimator.

    Args:
        basis_loglikelihoods (np.ndarray):
            The array of basis loglikelihood with the shape of (dim, n_basis, n_ei_candidates)

    Returns:
        config_loglikelihoods:
            The loglikelihoods of configs (n_ei_candidates, )

    TODO: Add test
    """
    # Product of kernels with respect to dimension
    config_basis_loglikelihoods = basis_loglikelihoods.sum(axis=0)
    # Compute config loglikelihoods using logsumexp to avoid overflow
    config_loglikelihoods = logsumexp(config_basis_loglikelihoods, b=weights[:, np.newaxis], axis=0)

    return config_loglikelihoods


def compute_config_loglikelihoods_ratio(basis_loglikelihoods_lower: np.ndarray,
                                        basis_loglikelihoods_upper: np.ndarray,
                                        weights_lower: np.ndarray, weights_upper: np.ndarray) -> np.ndarray:
    """ TODO: Add test and doc-string """
    config_ll_lower = compute_config_loglikelihoods(basis_loglikelihoods_lower, weights_lower)
    config_ll_upper = compute_config_loglikelihoods(basis_loglikelihoods_upper, weights_upper)
    return config_ll_lower - config_ll_upper
