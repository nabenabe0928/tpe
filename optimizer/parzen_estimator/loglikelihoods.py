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
    """
    # Product of kernels with respect to dimension
    config_basis_loglikelihoods = basis_loglikelihoods.sum(axis=0)
    # Compute config loglikelihoods using logsumexp to avoid overflow
    config_loglikelihoods = logsumexp(config_basis_loglikelihoods, b=weights[:, np.newaxis], axis=0)

    return config_loglikelihoods


def compute_config_loglikelihoods_ratio(basis_loglikelihoods_lower: np.ndarray,
                                        basis_loglikelihoods_upper: np.ndarray,
                                        weights_lower: np.ndarray, weights_upper: np.ndarray) -> np.ndarray:
    """
    A wrapper function for the computation of the loglikelihood ratio

    Args:
        basis_loglikelihoods_lower (np.ndarray):
            The basis loglikelihood values for the lower group.
            The shape is (n_lower_basis, n_samples).
        basis_loglikelihoods_upper (np.ndarray):
            The basis loglikelihood values for the upper group.
            The shape is (n_upper_basis, n_samples).
        weights_lower (np.ndarray):
            The weights for each basis in the lower group.
            The shape is (n_lower_basis, ).
        weights_upper (np.ndarray):
            The weights for each basis in the upper group.
            The shape is (n_upper_basis, ).

    Returns:
        config_ll_ratio (np.ndarray):
            The log of likelihood-ratio in the lower
            and the upper group for each sample.
            The shape is (n_samples)
    """
    config_ll_lower = compute_config_loglikelihoods(basis_loglikelihoods_lower, weights_lower)
    config_ll_upper = compute_config_loglikelihoods(basis_loglikelihoods_upper, weights_upper)
    return config_ll_lower - config_ll_upper
