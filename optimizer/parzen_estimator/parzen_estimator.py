from enum import IntEnum
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np

from optimizer.parzen_estimator.kernel import AitchisonAitkenKernel, GaussKernel, UniformKernel
from util.constants import CategoricalHPType, NumericType, NumericalHPType


class NumericalPriorType(IntEnum):
    uniform = 0
    gaussian = 1
    no_prior = 2

    def get_kernel(self, lb: NumericType, ub: NumericType,
                   q: Optional[NumericType] = None) -> GaussKernel:
        if lb == -np.inf and ub == np.inf:
            mu, sigma = 0.0, 1.0
        elif lb != -np.inf and ub != np.inf:
            mu, sigma = 0.5 * (lb + ub), ub - lb
        else:
            raise ValueError('Invalid domain for numerical parameters: [{}, {}]'.format(
                lb, ub
            ))

        if self.name == 'gaussian':
            return GaussKernel(mu=mu, sigma=sigma, lb=lb, ub=ub, q=q)
        elif self.name == 'uniform':
            return GaussKernel(mu=mu, sigma=100 * sigma, lb=lb, ub=ub, q=q)
        else:
            pt_names = [pt.name for pt in NumericalPriorType if pt.name != 'no_prior']
            raise ValueError(
                'prior name must be {}, but got {}.'.format(
                    pt_names,
                    self.name
                )
            )


class CategoricalPriorType(IntEnum):
    uniform = 0
    no_prior = 1

    def get_kernel(self, n_choices: int) -> UniformKernel:
        if self.name == 'uniform':
            return UniformKernel(n_choices=n_choices)
        else:
            pt_names = [pt.name for pt in CategoricalPriorType if pt.name != 'no_prior']
            raise ValueError(
                'prior name must be {}, but got {}.'.format(
                    pt_names,
                    self.name
                )
            )


class BaseParzenEstimator:
    _basis: List[Union[GaussKernel, AitchisonAitkenKernel, UniformKernel]]
    _weights: np.ndarray
    dtype: Type[np.number]

    def _sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        """
        The method to sample from the parzen estimator.
        This method needs to be wrapped by a child class.

        Args:
            rng (np.random.RandomState): The random seed
            n_samples (int): The number of samples

        Returns:
            samples (np.ndarray):
                Samples from the parzen estimator.
                The shape is (n_samples, ).
        """
        samples = [
            self.basis[active].sample(rng)
            for active in np.argmax(
                rng.multinomial(n=1, pvals=self.weights, size=n_samples),
                axis=-1
            )
        ]
        return np.array(samples, dtype=self.dtype)

    def basis_loglikelihood(self, xs: np.ndarray) -> np.ndarray:
        """
        Compute the kernel value for each basis in the parzen estimator.

        Args:
            xs (np.ndarray): The sampled values to compute each kernel value
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
        if len(xs.shape) > 1:
            raise ValueError('xs must be 1D array, but got {}D'.format(len(xs.shape)))

        return np.array([b.log_pdf(xs) for b in self.basis])

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def basis(self) -> List[Union[GaussKernel, AitchisonAitkenKernel, UniformKernel]]:
        return self._basis


class NumericalParzenEstimator(BaseParzenEstimator):
    def __init__(self, samples: np.ndarray, lb: NumericType, ub: NumericType,
                 weight_func: Callable, q: Optional[NumericType] = None,
                 dtype: Type[Union[np.number, int, float]] = np.float64,
                 prior: NumericalPriorType = NumericalPriorType.gaussian):

        self._lb, self._ub, self._q = lb, ub, q
        # TODO: Add tests
        dtype_choices = (np.int32, np.int64, np.float32, np.float64)
        if dtype is int:
            self.dtype = np.int32
        elif dtype is float:
            self.dtype = np.float64
        elif dtype in dtype_choices:
            self.dtype = dtype  # type: ignore
        else:
            raise ValueError('dtype for NumericalParzenEstimator must be {}, but got {}'.format(
                dtype_choices, dtype
            ))
        if np.any(samples < lb) or np.any(samples > ub):
            raise ValueError('All the samples must be in [{}, {}].'.format(lb, ub))

        self._weights, self._basis = self._calculate(
            samples=samples,
            weight_func=weight_func,
            prior=prior
        )

    def __repr__(self) -> str:
        ret = 'NumericalParzenEstimator(\n\tlb={}, ub={}, q={},\n'.format(
            self.lb, self.ub, self.q
        )
        for i, (w, b) in enumerate(zip(self.weights, self.basis)):
            ret += '\t({}) weight: {}, basis: {},\n'.format(i + 1, w, b)

        return ret + ')'

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        """ the wrapper method of _sample """
        samples = self._sample(rng=rng, n_samples=n_samples)
        return samples

    def _calculate(self, samples: np.ndarray, weight_func: Callable,
                   prior: NumericalPriorType) -> Tuple[np.ndarray, List[GaussKernel]]:
        """
        Calculate a bandwidth based on Scott's rule

        Args:
            samples (np.ndarray): Samples to use for the construction of
                                  the parzen estimator
            weight_func (Callable: int -> np.ndarray):
                The function to calculate the weights of each basis.
                The default is a uniform distribution.
            prior (NumericalPriorType): What prior to use.

        Returns:
            weights (np.ndarray): Weights for each basis
            basis (List[GaussKernel]):
                Kernel functions for the corresponding basis

        NOTE:
            The bandwidth is computed using the following reference:
                Scott, D.W. (1992) Multivariate Density Estimation:
                Theory, Practice, and Visualization.
        """
        domain_range = self.ub - self.lb
        no_prior = prior == NumericalPriorType.no_prior
        observed = samples if no_prior else np.append(samples, 0.5 * (self.lb + self.ub))
        weights = weight_func(observed.size)
        observed_std = observed.std(ddof=1)

        IQR = np.subtract.reduce(np.percentile(observed, [75, 25]))
        sigmas = 1.059 * min(IQR, observed_std) * observed.size ** (-0.2)
        # 99% of samples will be confined in mean \pm 0.025 * domain_range (2.5 sigma)
        sigmas = np.ones_like(samples) * np.clip(sigmas, 1e-2 * domain_range, 0.5 * domain_range)

        fixed_params = dict(lb=self.lb, ub=self.ub, q=self.q)
        basis = [GaussKernel(mu=mu, sigma=sigma, **fixed_params) for mu, sigma in zip(samples, sigmas)]

        if not no_prior:
            basis.append(prior.get_kernel(lb=self.lb, ub=self.ub, q=self.q))

        return weights, basis

    @property
    def lb(self) -> NumericType:
        return self._lb

    @property
    def ub(self) -> NumericType:
        return self._ub

    @property
    def q(self) -> Optional[NumericType]:
        return self._q


class CategoricalParzenEstimator(BaseParzenEstimator):
    def __init__(self, samples: np.ndarray, n_choices: int, weight_func: Callable,
                 top: float = 0.8, prior: CategoricalPriorType = CategoricalPriorType.uniform):

        if samples.dtype not in [np.int32, np.int64]:
            raise ValueError('samples for CategoricalParzenEstimator must be np.ndarray[np.int32/64], '
                             'but got {}.'.format(samples.dtype))
        if np.any(samples < 0) or np.any(samples >= n_choices):
            raise ValueError('All the samples must be in [0, n_choices).')

        self.dtype = np.int32
        self._n_choices, self._top = n_choices, top
        self._basis = [AitchisonAitkenKernel(n_choices=n_choices, choice=c, top=top) for c in samples]

        if prior != CategoricalPriorType.no_prior:
            self._basis.append(prior.get_kernel(n_choices=self.n_choices))

        self._weights = weight_func(samples.size + (prior != CategoricalPriorType.no_prior))

    def __repr__(self) -> str:
        ret = 'CategoricalParzenEstimator(\n\tn_choices={}, top={},\n'.format(
            self.n_choices, self.top
        )
        for i, (w, b) in enumerate(zip(self.weights, self.basis)):
            ret += '\t({}) weight: {}, basis: {},\n'.format(i + 1, w, b)

        return ret + ')'

    @property
    def n_choices(self) -> int:
        return self._n_choices

    @property
    def top(self) -> float:
        return self._top

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        """ the wrapper method of _sample """
        return self._sample(rng=rng, n_samples=n_samples)


def build_numerical_parzen_estimator(config: NumericalHPType, dtype: Type[Union[float, int]],
                                     vals: np.ndarray, weight_func: Callable,
                                     is_ordinal: bool) -> NumericalParzenEstimator:
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
        q, log, lb, ub = info.get('q', None), info['log'], info['lower'], info['upper']
    else:
        q, log, lb, ub = config.q, config.log, config.lower, config.upper

    if dtype is int or q is not None:
        if not log:
            if q is None:
                q = 1

            lb -= 0.5 * q
            ub += 0.5 * q
        else:
            q = None

    if log:
        dtype = float
        lb, ub = np.log(lb), np.log(ub)
        vals = np.log(vals)

    pe = NumericalParzenEstimator(samples=vals, lb=lb, ub=ub, weight_func=weight_func,
                                  q=q, dtype=dtype)

    return pe


def build_categorical_parzen_estimator(config: CategoricalHPType, vals: np.ndarray,
                                       weight_func: Callable) -> CategoricalParzenEstimator:
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
        raise ValueError('vals to build categorical parzen estimator must be '
                         'the list of symbols {}, but got the list of indices.'.format(choices))

    pe = CategoricalParzenEstimator(samples=choice_indices, n_choices=n_choices, weight_func=weight_func)

    return pe
