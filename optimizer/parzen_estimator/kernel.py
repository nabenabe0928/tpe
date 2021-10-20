from typing import Optional, Union

import numpy as np
from scipy.special import erf
from scipy.stats import truncnorm

from util.constants import EPS, NumericType, SQR2, SQR2PI


class GaussKernel():
    def __init__(self, mu: NumericType, sigma: NumericType,
                 lb: NumericType, ub: NumericType, q: Optional[NumericType] = None):
        """
        Attributes:
            mu: NumericType
                In general, this value is one of the observed values.
            sigma: NumericType
                Generally, it is called band width and there are so many methods to initialize this value.
            lb, ub, q: NumericType
                lower and upper bound and quantization value
            norm_const: NumericType
                The normalization constant of probability density function.
                In other words, when we integranl this kernel from lb to ub, we would obtain 1 as a result.
        """

        if mu < lb or mu > ub:
            raise ValueError('mu must be [{}, {}], but got {}'.format(lb, ub, mu))
        if sigma <= 0:
            raise ValueError('sigma must be non-negative, but got {}.'.format(sigma))

        self._initialized = False
        self._mu = mu
        self._sigma = max(sigma, EPS)
        self._init_domain_params(lb=lb, ub=ub, q=q)
        self._norm_const = 1.
        self._norm_const = 1. / (self.cdf(self.ub) - self.cdf(self.lb))
        self._logpdf_const = np.log(self.norm_const / (SQR2PI * self.sigma))

    def __repr__(self) -> str:
        return 'GaussKernel(lb={}, ub={}, q={}, mu={}, sigma={})'.format(
            self.lb, self.ub, self.q, self.mu, self.sigma
        )

    def _init_domain_params(
        self,
        lb: NumericType,
        ub: NumericType,
        q: Optional[NumericType] = None,
    ) -> None:

        if self._initialized:
            raise AttributeError("Cannot reset domain parameters.")

        if lb > ub:
            raise ValueError(
                'Lower bound lb for GaussKernel must be smaller than Upper bound ub, '
                'but got {:.4f} > {:.4f}.'.format(lb, ub)
            )

        self._initialized = True
        self._lb, self._ub, self._q = lb, ub, q

    def pdf(self, x: Union[NumericType, np.ndarray]) -> Union[NumericType, np.ndarray]:
        """
        Compute the probability density function values.

        Args:
            x (np.ndarray): Samples to compute the pdf

        Returns:
            pdf (np.ndarray):
                The probability density function value for each sample
        """

        if self.q is None:
            z = SQR2PI * self.sigma
            mahalanobis = ((x - self.mu) / self.sigma) ** 2
            return self.norm_const / z * np.exp(-0.5 * mahalanobis)
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x + 0.5 * self.q, self.lb))
            return integral_u - integral_l

    def log_pdf(self, x: Union[NumericType, np.ndarray]) -> Union[NumericType, np.ndarray]:
        """
        Compute the log of the probability density function values.
        Log prevents the possibility of underflow

        Args:
            x (np.ndarray): Samples to compute the log pdf

        Returns:
            log_pdf (np.ndarray):
                The log of the probability density function value
                for each sample
        """

        if self.q is None:
            mahalanobis = ((x - self.mu) / self.sigma) ** 2
            return self.logpdf_const - 0.5 * mahalanobis
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x + 0.5 * self.q, self.lb))
            """TODO: Check why i got zero division"""
            return np.log(integral_u - integral_l + EPS)

    def cdf(self, x: Union[NumericType, np.ndarray]) -> Union[NumericType, np.ndarray]:
        """
        Compute the cumulative density function values.

        Args:
            x (np.ndarray): Samples to compute the cdf

        Returns:
            cdf (np.ndarray):
                The cumulative density function value for each sample
                cdf[i] = integral[from -inf to x[i]] pdf(x') dx'
        """

        z = (x - self.mu) / (SQR2 * self.sigma)
        return self.norm_const * 0.5 * (1. + erf(z))

    def sample(self, rng: np.random.RandomState) -> NumericType:
        """
        Sample a value from the truncated Gauss kernel
            truncnorm.rvs(standarized lb, stadarized ub, scale=1.0)
            ==> val ~ N(0, 1.0) s.t. standarized lb <= val <= standarized ub

            Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html

        Args:
            rng (np.random.RandomState): The random state for numpy

        Returns:
            value (NumericType): A sampled value
        """
        trunc_lb, trunc_ub = (self.lb - self.mu) / self.sigma, (self.ub - self.mu) / self.sigma
        val = truncnorm.rvs(trunc_lb, trunc_ub, scale=1.0, random_state=rng)
        val = val * self.sigma + self.mu
        return val if self.q is None else np.round(val / self.q) * self.q

    @property
    def mu(self) -> NumericType:
        return self._mu

    @property
    def sigma(self) -> NumericType:
        return self._sigma

    @property
    def q(self) -> Optional[NumericType]:
        return self._q

    @property
    def lb(self) -> NumericType:
        return self._lb

    @property
    def ub(self) -> NumericType:
        return self._ub

    @property
    def norm_const(self) -> NumericType:
        return self._norm_const

    @property
    def logpdf_const(self) -> float:
        return self._logpdf_const


class CategoricalKernel():
    def cdf(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the probability of provided values.

        Args:
            x (np.ndarray): Samples to compute the pdf

        Returns:
            cdf (np.ndarray):
                The the probability for each sample
        """

        err_msg = "The choice must be between {} and {}, but got ".format(0, self.n_choices)

        if isinstance(x, int) and (x >= self.n_choices or x < 0):
            raise ValueError("{}{}.".format(err_msg, x))
        elif np.any(x < 0) or np.any(x >= self.n_choices):
            raise ValueError("{}{}.".format(err_msg, x))

        if isinstance(x, int):
            return self.top if x == self.choice else self.others
        else:
            probs = np.full_like(x, self.others, dtype=np.float32)
            probs[x == self.choice] = self.top
            return probs

    def pdf(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """ In categorical parameters, pdf == cdf """
        return self.cdf(x)

    def log_pdf(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the log of the probability density function values.

        Args:
            x (np.ndarray): Samples to compute the log pdf

        Returns:
            log_pdf (np.ndarray):
                The log of the probability density function value
                for each sample
        """
        return np.log(self.cdf(x))

    def sample(self, rng: np.random.RandomState) -> int:
        """
        Sample a categorical symbol from this kernel

        Args:
            rng (np.random.RandomState): The random state for numpy

        Returns:
            index (int): Index of a symbol
        """
        return int(rng.multinomial(n=1, pvals=self.probs, size=1).argmax())

    @property
    def choice(self) -> int:
        return self._choice  # type: ignore

    @property
    def n_choices(self) -> int:
        return self._n_choices  # type: ignore

    @property
    def top(self) -> float:
        return self._top  # type: ignore

    @property
    def others(self) -> float:
        return self._others  # type: ignore

    @property
    def probs(self) -> np.ndarray:
        return self._probs  # type: ignore


class AitchisonAitkenKernel(CategoricalKernel):
    def __init__(self, n_choices: int, choice: int, top: float):
        """
        The kernel for categorical parameters

        Attributes:
            n_choices (int):
                The number of choices.
            choice (int):
                The index of the target choice.
            top (float): (0., 1.)
                The probability that `choice` is taken.
                The hyperparameter that controls the exploration.
                Lower values lead to more exploration.
            others (float): (0., 1.)
                The probability that another choice is taken.
                This value is computed based on class variables.

        Title: The Aitchison and Aitken Kernel Function Revisited
        Reference: http://www.ccsenet.org/journal/index.php/jmr/article/download/24994/15579
        """

        if choice < 0 or choice >= n_choices:
            raise ValueError('choice must be in [0, n_choices), but got {}.'.format(choice))

        self._initialized = False
        self._init_domain_params(n_choices=n_choices, top=top)
        self._choice = choice
        self._probs = np.full(self.n_choices, self.others)
        self._probs[choice] = self.top

    def __repr__(self) -> str:
        return 'AitchisonAitkenKernel(n_choices={}, choice={}, top={}, others={}, probs={})'.format(
            self.n_choices, self.choice, self.top, self.others, self.probs
        )

    def _init_domain_params(self, n_choices: int, top: float) -> None:

        if self._initialized:
            raise AttributeError("Cannot reset domain parameters.")

        if top < 0 or top > 1:
            raise ValueError(
                'The top probability that `choice` is sampled must be '
                'between 0. and 1., but got {}.'.format(top)
            )

        if n_choices < 2:
            raise ValueError(
                'The number of choices `n_choice` must be larger than 1, '
                'but got {}.'.format(n_choices)
            )

        self._initialized = True
        self._n_choices, self._top = n_choices, top
        self._others = (1. - top) / (n_choices - 1)


class UniformKernel(CategoricalKernel):
    def __init__(self, n_choices: int):
        self._initialized = False
        self._init_domain_params(n_choices=n_choices)
        self._choice = 0
        self._probs = np.full(self.n_choices, self.others)
        self._probs[self.choice] = self.top

    def __repr__(self) -> str:
        return 'UniformKernel(n_choices={}, choice={}, top={}, others={}, probs={})'.format(
            self.n_choices, self.choice, self.top, self.others, self.probs
        )

    def _init_domain_params(self, n_choices: int) -> None:

        if self._initialized:
            raise AttributeError("Cannot reset domain parameters.")

        if n_choices < 2:
            raise ValueError(
                'The number of choices `n_choice` must be larger than 1, '
                'but got {}.'.format(n_choices)
            )

        self._initialized = True
        self._n_choices, self._top = n_choices, 1.0 / n_choices
        self._others = self._top
