from typing import Optional, Tuple, Type, Union

import numpy as np

from scipy.special import erf

from util.constants import CategoricalHPType, EPS, NumericType, NumericalHPType, SQR2PI, SQR2, uniform_weight


def calculate_norm_consts(
    lb: NumericType, ub: NumericType, means: np.ndarray, stds: np.ndarray
) -> Tuple[NumericType, NumericType]:
    """
    Args:
        lb (NumericType):
            The lower bound of a parameter.
        ub (NumericType):
            The upper bound of a parameter.
        means (np.ndarray):
            The mean value for each kernel basis. The shape is (n_samples, ).
        stds (np.ndarray):
            The bandwidth value for each kernel basis. The shape is (n_samples, ).

    Returns:
        norm_consts (NumericType):
            The normalization constant of each kernel due to the truncation.
        logpdf_consts (NumericType):
            The constant for loglikelihood computation.
    """
    zl = (lb - means) / (SQR2 * stds)
    zu = (ub - means) / (SQR2 * stds)
    norm_consts = 2.0 / (erf(zu) - erf(zl))
    logpdf_consts = np.log(norm_consts / (SQR2PI * stds))
    return norm_consts, logpdf_consts


class NumericalParzenEstimator:
    def __init__(
        self,
        samples: np.ndarray,
        lb: NumericType,
        ub: NumericType,
        q: Optional[NumericType] = None,
        dtype: Type[Union[np.number, int, float]] = np.float64,
        min_bandwidth_factor: float = 1e-2,
    ):

        self.lb, self.ub, self.q = lb, ub, q
        # TODO: Add tests
        dtype_choices = (np.int32, np.int64, np.float32, np.float64)
        self.dtype: Type[np.number]
        if dtype is int:
            self.dtype = np.int32
        elif dtype is float:
            self.dtype = np.float64
        elif dtype in dtype_choices:
            self.dtype = dtype  # type: ignore
        else:
            raise ValueError(f"dtype for NumericalParzenEstimator must be {dtype_choices}, but got {dtype}")
        if np.any(samples < lb) or np.any(samples > ub):
            raise ValueError(f"All the samples must be in [{lb}, {ub}].")

        self._calculate(samples=samples, min_bandwidth_factor=min_bandwidth_factor)

    def __repr__(self) -> str:
        ret = f"NumericalParzenEstimator(\n\tlb={self.lb}, ub={self.ub}, q={self.q},\n"
        for i, (w, m, s) in enumerate(zip(self.weights, self.means, self.stds)):
            ret += f"\t({i + 1}) weight: {w}, basis: GaussKernel(mean={m}, std={s}),\n"
        return ret + ")"

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the kernel value for each basis in the parzen estimator.

        Args:
            x (np.ndarray): The sampled values to compute each kernel value
                            The shape is (n_samples, )

        Returns:
            basis_loglikelihoods (np.ndarray):
                The kernel values for each basis given sampled values
                The shape is (B, n_samples)
                where B is the number of basis and n_samples = xs.size

        NOTE:
            When the parzen estimator is computed by:
                p(x) = sum[i = 0 to B] weights[i] * basis[i](x)
            where basis[i] is the i-th kernel function.
            Then this function returns the following:
                [log(basis[0](xs)), ..., log(basis[B - 1](xs))]
        """
        if self.q is None:
            mahalanobis = ((x - self.means[:, np.newaxis]) / self.stds[:, np.newaxis]) ** 2
            return self.logpdf_consts[:, np.newaxis] - 0.5 * mahalanobis
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x + 0.5 * self.q, self.lb))
            return np.log(integral_u - integral_l + EPS)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the cumulative density function values.

        Args:
            x (np.ndarray): Samples to compute the cdf

        Returns:
            cdf (np.ndarray):
                The cumulative density function value for each sample
                cdf[i] = integral[from -inf to x[i]] pdf(x') dx'
        """
        z = (x - self.means[:, np.newaxis]) / (SQR2 * self.stds[:, np.newaxis])
        return self.norm_consts[:, np.newaxis] * 0.5 * (1.0 + erf(z))

    def _sample(self, rng: np.random.RandomState, idx: int) -> NumericType:
        while True:
            val = rng.normal(loc=self.means[idx], scale=self.stds[idx])
            if self.lb <= val <= self.ub:
                return val if self.q is None else np.round(val / self.q) * self.q

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        samples = [
            self._sample(rng, active) for active in rng.choice(self.weights.size, p=self.weights, size=n_samples)
        ]
        return np.array(samples, dtype=self.dtype)

    def _calculate(self, samples: np.ndarray, min_bandwidth_factor: float) -> None:
        """
        Calculate parameters of KDE based on Scott's rule

        Args:
            samples (np.ndarray): Samples to use for the construction of
                                  the parzen estimator

        NOTE:
            The bandwidth is computed using the following reference:
                * Scott, D.W. (1992) Multivariate Density Estimation:
                  Theory, Practice, and Visualization.
                * Berwin, A.T. (1993) Bandwidth Selection in Kernel
                  Density Estimation: A Review. (page 12)
                * Nils, B.H, (2013) Bandwidth selection for kernel
                  density estimation: a review of fully automatic selector
                * Wolfgang, H (2005) Nonparametric and Semiparametric Models
        """
        domain_range = self.ub - self.lb
        means = np.append(samples, 0.5 * (self.lb + self.ub))  # Add prior at the end
        weights = uniform_weight(means.size)
        std = means.std(ddof=1)

        IQR = np.subtract.reduce(np.percentile(means, [75, 25]))
        bandwidth = 1.059 * min(IQR / 1.34, std) * means.size ** (-0.2)
        # 99% of samples will be confined in mean \pm 0.025 * domain_range (2.5 sigma)
        min_bandwidth = min_bandwidth_factor * domain_range
        clipped_bandwidth = np.ones_like(means) * np.clip(bandwidth, min_bandwidth, 0.5 * domain_range)
        clipped_bandwidth[-1] = domain_range  # The bandwidth for the prior

        self.means, self.stds = means, clipped_bandwidth
        self.norm_consts, self.logpdf_consts = calculate_norm_consts(
            lb=self.lb, ub=self.ub, means=self.means, stds=self.stds
        )
        self.weights = weights


class CategoricalParzenEstimator:
    def __init__(self, samples: np.ndarray, n_choices: int, top: float = 0.8):

        if samples.dtype not in [np.int32, np.int64]:
            raise ValueError(
                "samples for CategoricalParzenEstimator must be np.ndarray[np.int32/64], " f"but got {samples.dtype}."
            )
        if np.any(samples < 0) or np.any(samples >= n_choices):
            raise ValueError("All the samples must be in [0, n_choices).")

        self.dtype = np.int32
        self.n_choices = n_choices
        # AitchisonAitkenKernel: p = top or (1 - top) / (c - 1)
        # UniformKernel: p = 1 / c
        self.top, self.bottom, self.uniform = top, (1 - top) / (n_choices - 1), 1.0 / n_choices
        self.weight = uniform_weight(samples.size + 1)[0]
        indices, counts = np.unique(samples, return_counts=True)
        self.probs = np.full(n_choices, self.uniform)  # uniform prior, so the initial value is 1 / c.

        slicer = np.arange(n_choices)
        for idx, count in zip(indices, counts):
            self.probs[slicer != idx] += count * self.bottom
            self.probs[slicer == idx] += count * self.top

        self.probs *= self.weight
        likelihood_choices = np.array(
            [[self.top if i == j else self.bottom for j in range(n_choices)] for i in range(n_choices)]
        )
        bls = np.vstack([likelihood_choices[samples], np.full(n_choices, self.uniform)])
        self.basis_loglikelihoods = np.log(bls)  # shape = (n_basis, n_choices)

    def __repr__(self) -> str:
        return f"CategoricalParzenEstimator(n_choices={self.n_choices}, top={self.top}, probs={self.probs})"

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the kernel value for each basis in the parzen estimator.

        Args:
            x (np.ndarray): The sampled values to compute each kernel value
                            The shape is (n_samples, )

        Returns:
            basis_loglikelihoods (np.ndarray):
                The kernel values for each basis given sampled values
                The shape is (B, n_samples)
                where B is the number of basis and n_samples = xs.size

        NOTE:
            When the parzen estimator is computed by:
                p(x) = sum[i = 0 to B] weights[i] * basis[i](x)
            where basis[i] is the i-th kernel function.
            Then this function returns the following:
                [log(basis[0](xs)), ..., log(basis[B - 1](xs))]
        """
        return self.basis_loglikelihoods[:, x]

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        return rng.choice(self.n_choices, p=self.probs, size=n_samples)


def build_numerical_parzen_estimator(
    config: NumericalHPType,
    dtype: Type[Union[float, int]],
    vals: np.ndarray,
    is_ordinal: bool,
    min_bandwidth_factor: float = 1e-2,
) -> NumericalParzenEstimator:
    """
    Build a numerical parzen estimator

    Args:
        config (NumericalHPType): Hyperparameter information from the ConfigSpace
        dtype (Type[np.number]): The data type of the hyperparameter
        vals (np.ndarray): The observed hyperparameter values
        is_ordinal (bool): Whether the configuration is ordinal

    Returns:
        pe (NumericalParzenEstimator): Parzen estimator given a set of observations
    """

    if is_ordinal:
        info = config.meta
        q, log, lb, ub = info.get("q", None), info.get("log", False), info["lower"], info["upper"]
    else:
        q, log, lb, ub = config.q, config.log, config.lower, config.upper

    if dtype is int or q is not None:
        if log:
            q = None
        elif q is None:
            q = 1
        if q is not None:
            lb -= 0.5 * q
            ub += 0.5 * q

    if log:
        dtype = float
        lb, ub = np.log(lb), np.log(ub)
        vals = np.log(vals)

    pe = NumericalParzenEstimator(
        samples=vals, lb=lb, ub=ub, q=q, dtype=dtype, min_bandwidth_factor=min_bandwidth_factor
    )

    return pe


def build_categorical_parzen_estimator(config: CategoricalHPType, vals: np.ndarray) -> CategoricalParzenEstimator:
    """
    Build a categorical parzen estimator

    Args:
        config (CategoricalHPType): Hyperparameter information from the ConfigSpace
        vals (np.ndarray): The observed hyperparameter values (i.e. symbols, but not indices)

    Returns:
        pe (CategoricalParzenEstimator): Parzen estimators given a set of observations
    """
    choices = config.choices
    n_choices = len(choices)

    try:
        choice_indices = np.array([choices.index(val) for val in vals], dtype=np.int32)
    except ValueError:
        raise ValueError(
            "vals to build categorical parzen estimator must be "
            f"the list of symbols {choices}, but got the list of indices."
        )

    pe = CategoricalParzenEstimator(samples=choice_indices, n_choices=n_choices)

    return pe
