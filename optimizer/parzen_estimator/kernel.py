import numpy as np
from scipy.special import erf


EPS = 1.0e-12


class GaussKernel():
    def __init__(self, mu, sigma, lb, ub, q):
        """
        The hyperparameters of Gauss Kernel.

        mu: float
            In general, this value is one of the observed values.
        sigma: float
            Generally, it is called band width and there are so many methods to initialize this value.
        lb, ub, q: float or int
            lower and upper bound and quantization value
        weight: float
            The normalization constant of probability density function.
            In other words, when we integranl this kernel from lb to ub, we would obtain 1 as a result.
        """

        self.mu = mu
        self.sigma = max(sigma, EPS)
        self.lb, self.ub, self.q = lb, ub, q
        self.weight = 1.
        self.weight = 1. / (self.cdf(ub) - self.cdf(lb))

    def pdf(self, x):
        """
        Returning the value Probability density function of a given x.
        """

        if self.q is None:
            z = np.sqrt(2 * np.pi) * self.sigma
            mahalanobis = ((x - self.mu) / self.sigma) ** 2
            return self.weight / z * np.exp(-0.5 * mahalanobis)
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x + 0.5 * self.q, self.lb))
            return integral_u - integral_l

    def cdf(self, x):
        """
        Returning the value of Cumulative distribution function at a given x.
        """

        z = (x - self.mu) / (np.sqrt(2) * self.sigma)
        return self.weight * 0.5 * (1. + erf(z))

    def sample_from_kernel(self, rng):
        """
        Returning the random number sampled from this Gauss kernel.
        """

        while True:
            sample = rng.normal(loc=self.mu, scale=self.sigma)
            if self.lb <= sample <= self.ub:
                return sample


class AitchisonAitkenKernel():
    def __init__(self, choice, n_choices, top=0.9):
        """
        Reference: http://www.ccsenet.org/journal/index.php/jmr/article/download/24994/15579

        Hyperparameters of Aitchison Aitken Kernel.

        n_choices: int
            The number of choices.
        choice: int
            The ID of the target choice.
        top: float (0. to 1.)
            The hyperparameter controling the extent of the other choice's distribution.
        """

        self.n_choices = n_choices
        self.choice = choice
        self.top = top

    def cdf(self, x):
        """
        Returning a probability of a given x.
        """

        if x == self.choice:
            return self.top
        elif 0 <= x <= self.n_choices - 1:
            return (1. - self.top) / (self.n_choices - 1)
        else:
            raise ValueError("The choice must be between {} and {}, but {} was given.".format(0, self.n_choices - 1, x))

    def cdf_for_numpy(self, xs):
        """
        Returning probabilities of a given list x.
        """

        return_val = np.array([])
        for x in xs:
            np.append(return_val, self.cdf(x))
        return return_val

    def probabilities(self):
        """
        Returning probabilities of every possible choices.
        """

        return np.array([self.cdf(n) for n in range(self.n_choices)])

    def sample_from_kernel(self, rng):
        """
        Returning random choice sampled from this Kernel.
        """

        choice_one_hot = rng.multinomial(n=1, pvals=self.probabilities(), size=1)
        return np.dot(choice_one_hot, np.arange(self.n_choices))
