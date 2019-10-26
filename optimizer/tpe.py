import numpy as np
import utils
from optimizer.parzen_estimator import NumericalParzenEstimator, CategoricalParzenEstimator
from optimizer import BaseOptimizer


EPS = 1e-12


def default_gamma(x, n_samples_lower=25):

    return min(int(np.ceil(0.25 * np.sqrt(x))), n_samples_lower)


def default_weights(x, n_samples_lower=25):
    if x == 0:
        return np.asarray([])
    elif x < n_samples_lower:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - n_samples_lower)
        flat = np.ones(n_samples_lower)
        return np.concatenate([ramp, flat], axis=0)


class TPE(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 n_parallels=1,
                 n_init=10,
                 max_evals=100,
                 n_experiments=0,
                 restart=True,
                 seed=None,
                 verbose=True,
                 print_freq=1,
                 n_ei_candidates=24,
                 rule="james",
                 gamma_func=default_gamma,
                 weight_func=default_weights):
        """
        n_ei_candidates: int
            The number of points to evaluate the EI function.
        gamma_func: callable
            The function returning the number of a better group based on the total number of evaluations.
        weight_func: callable
            The function returning the coefficients of each kernel.
        """

        super().__init__(hp_utils,
                         n_parallels=n_parallels,
                         n_init=n_init,
                         max_evals=max_evals,
                         n_experiments=n_experiments,
                         restart=restart,
                         seed=seed,
                         verbose=verbose,
                         print_freq=print_freq
                         )

        self.n_ei_candidates = n_ei_candidates
        self.gamma_func = gamma_func
        self.weight_func = weight_func
        self.opt = self.sample
        self.rule = rule

    def sample(self):
        hps_conf, _ = self.hp_utils.load_hps_conf(convert=True, do_sort=True, index_from_conf=False)
        hp_conf = []

        for idx, hps in enumerate(hps_conf):
            n_lower = self.gamma_func(len(hps))
            lower_vals, upper_vals = hps[:n_lower], hps[n_lower:]
            var_name = self.hp_utils.config_space._idx_to_hyperparameter[idx]
            var_type = utils.distribution_type(self.hp_utils.config_space, var_name)

            if var_type in [float, int]:
                hp_value = self._sample_numerical(var_name, var_type, lower_vals, upper_vals)
            else:
                hp_value = self._sample_categorical(var_name, lower_vals, upper_vals)
            hp_conf.append(hp_value)

        return self.hp_utils.revert_hp_conf(hp_conf)

    def _sample_numerical(self, var_name, var_type, lower_vals, upper_vals):
        """
        Parameters
        ----------
        lower_vals: ndarray (N_lower, )
            The values of better group.
        upper_vals: ndarray (N_upper, )
            The values of worse group.
        var_name: str
            The name of a hyperparameter
        var_type: type
            The type of a hyperparameter
        """

        hp = self.hp_utils.config_space._hyperparameters[var_name]
        q, log, lb, ub, converted_q = hp.q, hp.log, 0., 1., None

        if var_type is int or q is not None:
            if not log:
                converted_q = 1. / (hp.upper - hp.lower) if q is None else q / (hp.upper - hp.lower)
                lb -= 0.5 * converted_q
                ub += 0.5 * converted_q

        pe_lower = NumericalParzenEstimator(lower_vals, lb, ub, self.weight_func, q=converted_q, rule=self.rule)
        pe_upper = NumericalParzenEstimator(upper_vals, lb, ub, self.weight_func, q=converted_q, rule=self.rule)

        """
        from optimizer.parzen_estimator import plot_density_estimators
        plot_density_estimators(pe_lower, pe_upper, var_name, pr_basis=True, pr_basis_mu=True)
        """

        return self._compare_candidates(pe_lower, pe_upper)

    def _sample_categorical(self, var_name, lower_vals, upper_vals):
        choices = self.hp_utils._hyperparameters[var_name].choices
        n_choices = len(choices)
        lower_vals = [choices.index(val) for val in lower_vals]
        upper_vals = [choices.index(val) for val in upper_vals]

        pe_lower = CategoricalParzenEstimator(lower_vals, n_choices, self.weight_func)
        pe_upper = CategoricalParzenEstimator(upper_vals, n_choices, self.weight_func)

        best_choice_idx = int(self._compare_candidates(pe_lower, pe_upper))

        return choices[best_choice_idx]

    def _compare_candidates(self, pe_lower, pe_upper):
        samples_lower = pe_lower.sample_from_density_estimator(self.rng, self.n_ei_candidates)
        best_idx = np.argmax(pe_lower.log_likelihood(samples_lower) - pe_upper.log_likelihood(samples_lower))
        return samples_lower[best_idx]
