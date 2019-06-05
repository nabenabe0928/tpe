import numpy as np
import scipy.special
import csv
import random
import ConfigSpace as CS
import ConfigSpace as CSH
from sampler.parzen_estimator import ParzenEstimator
from sampler.parzen_estimator import ParzenEstimatorParameters

EPS = 1e-12

def get_evaluations(model, num):
    with open("evaluation/{}/evaluation{:0>3}.csv".format(model, num), "r", newline = "") as f:

        reader = list(csv.DictReader(f, delimiter = ",", quotechar = '"'))
        param_names = list(reader[0].keys())
        
        losses = []
        hyperparameters = {param_name :[] for param_name in param_names if param_name != "loss"}

        for row in reader:
            for param_name in param_names:
                if param_name == "loss":
                    losses.append(float(row[param_name]))
                else:
                    try:
                        hyperparameters[param_name].append(eval(row[param_name]))
                    except:
                        hyperparameters[param_name].append(row[param_name])
    
    hyperparameters = {name: np.array(hps) for name, hps in hyperparameters.items()}
    return hyperparameters, losses

def distribution_type(cs, var_name):    
    cs_dist = str(type(cs._hyperparameters[var_name]))

    if "Integer" in cs_dist:
        return "int"
    elif "Float" in cs_dist:
        return "float"
    elif "Categorical" in cs_dist:
        return "cat"
    else:
        raise NotImplementedError("The distribution is not implemented.")

def default_gamma(x, n_samples_lower = 25):

    return min(int(np.ceil(0.25 * np.sqrt(x))), n_samples_lower)


def default_weights(x, n_samples_lower = 25):
    if x == 0:
        return np.asarray([])
    elif x < n_samples_lower:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num = x - n_samples_lower)
        flat = np.ones(n_samples_lower)
        return np.concatenate([ramp, flat], axis = 0)

class TPESampler():
    def __init__(self, model, num, config_space, n_jobs, consider_prior = True, prior_weight = 1.0,
            consider_magic_clip = True, consider_endpoints = False, n_startup_trials = 10,
            n_ei_candidates = 24, gamma_func = default_gamma, weight_func = default_weights
        ):
        
        self.config_space = config_space
        self.hyperparameters, self.losses = get_evaluations(model, num)

        self.parzen_estimator_parameters = ParzenEstimatorParameters(
            consider_prior, prior_weight, consider_magic_clip, consider_endpoints, weight_func)
        self.prior_weight = prior_weight
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma_func = gamma_func
        self.weight_func = weight_func
        self.rng = np.random.RandomState()
        
    def sample(self):
        n = len(self.losses)

        if n < self.n_startup_trials:
            sample_dict = self.config_space.sample_configuration().get_dictionary()

            for var_name in self.config_space.keys():
                hp = self.config_space._hyperparameters[var_name]
                q = hp.q
                if q is not None and hp.is_log:
                    rnd = random.random() * (hp.upper - hp.lower) + hp.lower
                    sample_dict[var_name] = np.exp(np.round(rnd / q) * q)

            return sample_dict

        sample_dict = {var_name : None for var_name in self.hyperparameters.keys()}

        for var_name in self.hyperparameters.keys():
            lower_vals, upper_vals = self._split_observation_pairs(var_name)

            _dist = distribution_type(self.config_space, var_name)
            q = self.config_space._hyperparameters[var_name].q

            if _dist is "cat":
                cat_idx = sample_dict[var_name] = self._sample_categorical(var_name, lower_vals, upper_vals)
                sample_dict[var_name] = self.config_space._hyperparameters[var_name].choices[cat_idx]
            elif _dist is "float" or _dist is "int":
                sample_dict[var_name] = self._sample_numerical(_dist, var_name, lower_vals, upper_vals, q = q)
        
        """
        #####
        for var_name in self.hyperparameters.keys():
            hoge = None
            cond = None
            if hoge == "cond" and not cond:
                sample_dict[var_name] = None
        ##### here, i have to add the function to make the conditional parameters None to prevent the conditional parameters 
        ##### density estimator including the point which was not used to evaluate the algorithm.
        """

        return sample_dict

    def _split_observation_pairs(self, var_name):
        observation_pairs = [[hp, loss] for hp, loss in zip(self.hyperparameters[var_name], self.losses) if hp != None]
        config_vals, loss_vals = np.asarray([p[0] for p in observation_pairs]), np.asarray([p[1] for p in observation_pairs])
        
        n_lower = self.gamma_func(len(config_vals))
        loss_ascending = np.argsort(loss_vals)
        
        lower_vals = np.asarray(config_vals[loss_ascending[:n_lower]])
        upper_vals = np.asarray(config_vals[loss_ascending[n_lower:]])

        return lower_vals, upper_vals

    def _sample_numerical(self, _dist, var_name, lower_vals, upper_vals, q = None):

        lower_bound, upper_bound = self.config_space._hyperparameters[var_name].lower, self.config_space._hyperparameters[var_name].upper
        is_log = self.config_space._hyperparameters[var_name].log

        if is_log:
            lower_bound = np.log(lower_bound)
            upper_bound = np.log(upper_bound)
            lower_vals = np.log(lower_vals)
            upper_vals = np.log(upper_vals)
        
        if _dist == "int":
            lower_bound -= 0.5
            upper_bound += 0.5
        elif q is not None:
            if is_log:
                lower_bound -= 0.5 * q
                upper_bound += 0.5 * q
            else:
                lower_bound -= 0.5 * q
                upper_bound += 0.5 * q

        size = (self.n_ei_candidates, )

        parzen_estimator_lower = ParzenEstimator(lower_vals, lower_bound, upper_bound, self.parzen_estimator_parameters)
        samples_lower = self._sample_from_gmm(parzen_estimator_lower, lower_bound, upper_bound, _dist, size = size, is_log = is_log, q = q)
        log_likelihoods_lower = self._gmm_log_pdf(samples_lower, parzen_estimator_lower, lower_bound, upper_bound, var_type = _dist, is_log=is_log, q = q)

        parzen_estimator_upper = ParzenEstimator(upper_vals, lower_bound, upper_bound, self.parzen_estimator_parameters)

        log_likelihoods_upper = self._gmm_log_pdf(samples_lower, parzen_estimator_upper, lower_bound, upper_bound, var_type = _dist, is_log = is_log, q = q)

        return eval(_dist)(TPESampler._compare(samples_lower, log_likelihoods_lower, log_likelihoods_upper))

    def _sample_categorical(self, var_name, lower_vals, upper_vals):

        choices = self.config_space._hyperparameters[var_name].choices
        lower_vals = [choices.index(val) for val in lower_vals]
        upper_vals = [choices.index(val) for val in upper_vals]

        n_choices = len(choices)
        size = (self.n_ei_candidates, )

        weights_lower = self.weight_func(len(lower_vals))
        counts_lower = np.bincount(lower_vals, minlength = n_choices, weights = weights_lower)
        weighted_lower = counts_lower + self.prior_weight
        weighted_lower /= weighted_lower.sum()
        samples_lower = self._sample_from_categorical_dist(weighted_lower, size = size)
        log_likelihoods_lower = TPESampler._categorical_log_pdf(samples_lower, weighted_lower)

        weights_upper = self.weight_func(len(upper_vals))
        counts_upper = np.bincount(upper_vals, minlength = n_choices, weights = weights_upper)
        weighted_upper = counts_upper + self.prior_weight
        weighted_upper /= weighted_upper.sum()
        log_likelihoods_upper = TPESampler._categorical_log_pdf(samples_lower, weighted_upper)

        return int(TPESampler._compare(samples_lower, log_likelihoods_lower, log_likelihoods_upper))

    def _sample_from_gmm(self, parzen_estimator, lower, upper, var_type, size=(), is_log = False, q = None):
        
        weights = parzen_estimator.weights
        mus = parzen_estimator.mus
        sigmas = parzen_estimator.sigmas
        weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
        n_samples = np.prod(size)

        if lower >= upper:
            raise ValueError("The 'lower' should be lower than the 'upper'. "
                             "But (lower, upper) = ({}, {}).".format(lower, upper))
        samples = np.asarray([], dtype=float)
        while samples.size < n_samples:
            active = np.argmax(self.rng.multinomial(1, weights))
            draw = self.rng.normal(loc = mus[active], scale = sigmas[active])
            if lower <= draw < upper:
                samples = np.append(samples, draw)

        if var_type == "float" and q is None:
            if is_log:
                samples = np.exp(samples)
            return samples
        else:
            q = q if q is not None else 1
            if is_log:
                samples = np.exp(samples)
            return np.round(samples / q) * q

    def _gmm_log_pdf(self, samples, parzen_estimator, lower, upper, var_type, is_log = False, q = None):
        
        weights = parzen_estimator.weights
        mus = parzen_estimator.mus
        sigmas = parzen_estimator.sigmas
        samples, weights, mus, sigmas = map(np.asarray, (samples, weights, mus, sigmas))
        p_accept = np.sum(weights * (TPESampler._normal_cdf(upper, mus, sigmas) - TPESampler._normal_cdf(lower, mus, sigmas)))

        if var_type == "float" and q is None:
            jacobian_inv = samples[:, None] if is_log else np.ones(samples.shape)[:, None]
            if is_log:
                distance = np.log(samples[:, None]) - mus
            else:
                distance = samples[:, None] - mus
            mahalanobis = (distance / np.maximum(sigmas, EPS))**2
            Z = np.sqrt(2 * np.pi) * sigmas * jacobian_inv
            coef = weights / Z / p_accept
            return_val = TPESampler._logsum_rows(-0.5 * mahalanobis + np.log(coef))
        else:
            q = q if q is not None else 1
            probabilities = np.zeros(samples.shape, dtype = float)
            cdf = TPESampler._log_normal_cdf if is_log else TPESampler._normal_cdf
            for w, mu, sigma in zip(weights, mus, sigmas):
                if is_log:
                    upper_bound = np.minimum(samples + 0.5 * q, np.exp(upper))
                    lower_bound = np.maximum(samples - 0.5 * q, np.exp(lower))
                    lower_bound = np.maximum(0, lower_bound)
                else:
                    upper_bound = np.minimum(samples + 0.5 * q, upper)
                    lower_bound = np.maximum(samples - 0.5 * q, lower)
                probabilities += w * (cdf(upper_bound, mu, sigma) - cdf(lower_bound, mu, sigma))
            return_val = np.log(probabilities + EPS) - np.log(p_accept + EPS)

        return_val.shape = samples.shape
        
        return return_val

    def _sample_from_categorical_dist(self, probabilities, size=()):
        
        if probabilities.size == 1 and isinstance(probabilities[0], np.ndarray):
            probabilities = probabilities[0]
        probabilities = np.asarray(probabilities)

        if size == ():
            size = (1, )
        elif isinstance(size, (int, np.number)):
            size = (size, )
        else:
            size = tuple(size)

        if size == (0, ):
            return np.asarray([], dtype=float)

        n_draws = int(np.prod(size))
        sample = self.rng.multinomial(n = 1, pvals = probabilities, size = int(n_draws))
        return_val = np.dot(sample, np.arange(probabilities.size))
        return_val.shape = size
        return return_val

    @classmethod
    def _categorical_log_pdf(cls, samples, p):
        
        if samples.size:
            return np.log(np.asarray(p)[samples])
        else:
            return np.asarray([])

    @classmethod
    def _compare(cls, samples, log_lower, log_upper):
        
        samples, log_lower, log_upper = map(np.asarray, (samples, log_lower, log_upper))
        score = log_lower - log_upper
        best = np.argmax(score)
        
        return samples[best]
        
    @classmethod
    def _logsum_rows(cls, x):

        x = np.asarray(x)
        return np.log(np.exp(x).sum(axis = 1))
        
    @classmethod
    def _normal_cdf(cls, x, mu, sigma):
        mu, sigma = map(np.asarray, (mu, sigma))
        denominator = x - mu
        numerator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = denominator / numerator
        return 0.5 * (1 + scipy.special.erf(z))

    @classmethod
    def _log_normal_cdf(cls, x, mu, sigma):
        mu, sigma = map(np.asarray, (mu, sigma))
        
        numerator = np.log(np.maximum(x, EPS)) - mu
        denominator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = numerator / denominator
        return .5 + .5 * scipy.special.erf(z)

