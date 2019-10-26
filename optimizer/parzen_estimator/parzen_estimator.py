import numpy as np
from optimizer.parzen_estimator import GaussKernel, AitchisonAitkenKernel


EPS = 1e-12


def plot_density_estimators(pe_lower, pe_upper, var_name, pr_basis=False, pr_ei=False, pr_basis_mu=False):
    import matplotlib.pyplot as plt

    weights_set = [pe_lower.weights, pe_upper.weights]
    basis_set = [pe_lower.basis, pe_upper.basis]
    mus_set = [pe_lower.mus, pe_upper.mus]
    names = ["lower", "upper"]
    lb, ub = pe_lower.lb, pe_lower.ub
    cmap = plt.get_cmap("tab10")

    x = np.linspace(lb, ub, 100)
    des = np.array([np.zeros(100) for _ in range(2)])

    for i, (weights, basis, mus, de, name) in enumerate(zip(weights_set, basis_set, mus_set, des, names)):
        for w, b, mu in zip(weights, basis, mus):
            de += w * b.pdf(x)
            if pr_basis:
                plt.plot(x, w * b.pdf(x), color=cmap(i), linestyle="dotted")
            if pr_basis_mu:
                plt.plot([mu] * 100, np.linspace(0, w, 100), color=cmap(i), linestyle="dotted")
        plt.plot(x, de, label=name, color=cmap(i))

    if pr_ei:
        plt.plot(x, np.log(des[0]) - np.log(des[1]), label="EI function", color=cmap(2))

    plt.title("Parzen Estimators for {} with {} lower and {} upper evaluations.".format(var_name, len(mus_set[0]) - 1, len(mus_set[1]) - 1))
    plt.xlim(lb, ub)
    plt.grid()
    plt.legend()
    plt.show()


class NumericalParzenEstimator(object):
    """
    samples: ndarray (n, )
        The observed hyperparameter values.
    lb: float
        The lower bound of a hyperparameter
    ub: float
        The upper bound of a hyperparameter
    weight_func: callable
        The function returning the weight of each basis of this parzen estimator.
    q: float
        The quantization value.
    """

    def __init__(self, samples, lb, ub, weight_func, q=None, rule="james"):
        """
        Here, the number of basis is n + 1.
        n basis are from observed values and 1 basis is from the prior distribution which is N((lb + ub) / 2, (ub - lb) ** 2).

        if james rule...
            weights: ndarray (n + 1, ) 
                the weight of each basis. The total must be 1.
            mus: ndarray (n + 1, )
                The center of each basis.
                The values themselves are the observed hyperparameter values. Sorted in ascending order.
            sigmas: ndarray (n + 1, )
                The band width of each basis.
                The values are determined by a heuristic.
            basis: the list of kernel.GaussKernel object (n + 1, )
        if scott rule...
            weights: ndarray (n, ) 
                the weight of each basis. The total must be 1.
            mus: ndarray (n, )
                The center of each basis.
            sigmas: ndarray (n, )
                The band width of each basis.
                The values are determined by a scott rule.
            basis: the list of kernel.GaussKernel object (n, )
        """

        self.lb, self.ub, self.q, self.rule = lb, ub, q, rule
        self.weights, self.mus, self.sigmas = self._calculate(samples, weight_func)
        self.basis = [GaussKernel(m, s, lb, ub, q) for m, s in zip(self.mus, self.sigmas)]

    def sample_from_density_estimator(self, rng, n_samples):
        """
        Parameters
        ----------
        rng: numpy.random.RandomState object
        n_samples: int
            The number of samples

        Returns
        -------
        samples: ndarray (n_samples, )
            The random number sampled from the parzen estimator.
        """

        samples = np.asarray([], dtype=float)
        while samples.size < n_samples:
            active = np.argmax(rng.multinomial(1, self.weights))
            drawn_hp = self.basis[active].sample_from_kernel(rng)
            samples = np.append(samples, drawn_hp)

        return samples if self.q is None else np.round(samples / self.q) * self.q

    def log_likelihood(self, xs):
        """
        Parameters
        ----------
        xs: ndarray (n_ei_candidates, )
            The number of candidates to evaluate the EI function.

        Returns
        -------
        loglikelihood of the parzen estimator at given points: ndarray (n_ei_candidates, )
            Here, we do not consider jacobian, because it will be canceled out when we compute EI function.
        """

        ps = np.zeros(xs.shape, dtype=float)
        for w, b in zip(self.weights, self.basis):
            ps += w * b.pdf(xs)

        return np.log(ps + EPS)

    def _calculate(self, samples, weights_func):
        if self.rule == "james":
            return self._calculate_by_james_rule(samples, weights_func)
        elif self.rule == "scott":
            return self._calculate_by_scott_rule(samples)
        else:
            raise ValueError("Rule must be 'scott' or 'james'.")
    
    def _calculate_by_james_rule(self, samples, weights_func):
        samples = np.asarray(samples)
        prior_mu = 0.5 * (self.lb + self.ub)
        sigma_bounds = [(self.ub - self.lb) / min(100.0, (1.0 + samples.size)), self.ub - self.lb]

        order = np.argsort(samples)
        sorted_mus = samples[order]
        prior_pos = np.searchsorted(samples[order], prior_mu)
        sorted_mus = np.insert(sorted_mus, prior_pos, prior_mu)

        sorted_mus_with_bounds = np.insert([sorted_mus[0], sorted_mus[-1]], 1, sorted_mus)
        sigmas = np.maximum(sorted_mus_with_bounds[1:-1] - sorted_mus_with_bounds[0:-2], sorted_mus_with_bounds[2:] - sorted_mus_with_bounds[1:-1])
        sigmas = np.clip(sigmas, sigma_bounds[0], sigma_bounds[1])
        sigmas[prior_pos] = sigma_bounds[1]

        sorted_weights = weights_func(samples.size)[order]
        sorted_weights = np.insert(sorted_weights, prior_pos, 1.)
        sorted_weights /= sorted_weights.sum()

        return np.array(sorted_weights), np.array(sorted_mus), np.array(sigmas)

    def _calculate_by_scott_rule(self, samples):
        samples = np.array(samples)
        mus = np.append(samples, 0.5 * (self.lb + self.ub))
        mus_sigma = mus.std(ddof=1)
        IQR = np.subtract.reduce(np.percentile(mus, [75, 25]))
        sigma = 1.059 * min(IQR, mus_sigma) * mus.size ** (-0.2)
        sigmas = np.ones(mus.size) * np.clip(sigma, 1.0e-2 * (self.ub - self.lb), 0.5 * (self.ub - self.lb))
        sigmas[-1] = self.ub - self.lb
        weights = np.ones(mus.size)
        weights /= weights.sum()

        return weights, mus, sigmas


class CategoricalParzenEstimator():
    def __init__(self, samples, n_choices, weights_func, top=0.9):
        self.n_choices = n_choices
        self.mus = samples
        self.basis = [AitchisonAitkenKernel(c, n_choices, top=top) for c in samples]
        self.weights = weights_func(len(samples))
        self.weights /= self.weights.sum()

    def sample_from_density_estimator(self, rng, n_samples):
        basis_samples = rng.multinomial(n=1, pvals=self.weights, size=n_samples)
        basis_idxs = np.dot(basis_samples, np.arange(self.weights.size))
        return np.array([self.basis[idx].sample_from_kernel(rng) for idx in basis_idxs])

    def log_likelihood(self, samples):
        ps = np.zeros(samples.shape, dtype=float)
        for w, b in zip(self.weights, self.basis):
            ps += w * b.cdf_for_numpy(samples)
        return np.log(ps + EPS)
