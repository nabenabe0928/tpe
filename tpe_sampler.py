import numpy as np
import scipy.special
import csv
import ConfigSpace as CS
import ConfigSpace as CSH
from parzen_estimator import ParzenEstimator
from parzen_estimator import ParzenEstimatorParameters


EPS = 1e-12

def get_evaluations(model, num):
    with open("evaluation/{}/{}/evaluation.csv".format(model, num), "r", newline = "") as f:
        reader = dict(csv.DictReader(f, delimieter = ",", quotechar = '"'))
        param_names = list(csv.DictReader(f, delimieter = ",", quotechar = '"').fieldnames)
        losses = []
        hyperparameters = {param_name :[] for param_name in param_names if param_name != "loss"}

        for row in reader:
            for param_name in param_names:
                if param_name == "loss":
                    losses.append(row[param_name])
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

def get_variable_features(model):
    with open("feature/{}/feature.csv".format(model), "r", newline = "") as f:
        reader = dict(csv.DictReader(f, delimieter = ";", quotechar = '"'))
        config_space = CS.ConfigurationSpace()
        
        for row in reader:
            default = eval(row["default"])
            param_name = row["var_name"]
            var_type = row["type"]
            dist = row["dist"]

            if var_type == "cat":
                choices = eval(row["bound"])
                hp = CSH.CategoricalHyperparameter(name = param_name, choices = choices, default_value = default)
            else:
                b = eval(row["bound"])

                if var_type == "int":
                    if dist == "log":
                        hp = CSH.UniformIntegerHyperparameter(name = row["var_name"], lower = b[0], upper = b[1], default_value = default, log = True)
                    else:
                        hp = CSH.UniformIntegerHyperparameter(name = row["var_name"], lower = b[0], upper = b[1], default_value = default, log = False)
                elif var_type == "float":
                    if dist == "log":
                        hp = CSH.UniformFloatHyperparameter(name = row["var_name"], lower = b[0], upper = b[1], default_value = default, log = True)
                    else:
                        hp = CSH.UniformFloatHyperparameter(name = row["var_name"], lower = b[0], upper = b[1], default_value = default, log = False)
            
            config_space.add_hyperparameter(hp)
    
    return config_space

def default_gamma(x, n_samples_lower = 25):
    # type: (int) -> int

    return min(int(np.ceil(0.25 * np.sqrt(x))), n_samples_lower)


def default_weights(x, n_samples_lower = 25):
    # type: (int) -> np.ndarray

    if x == 0:
        return np.asarray([])
    elif x < n_samples_lower:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - n_samples_lower)
        flat = np.ones(n_samples_lower)
        return np.concatenate([ramp, flat], axis=0)

class TPESampler():
    def __init__(
            self,
            model,
            num,
            consider_prior = True,
            prior_weight = 1.0,
            consider_magic_clip = True,
            consider_endpoints = False,
            n_startup_trials = 10,
            n_ei_candidates = 24, 
            gamma = default_gamma,  # type: Callable[[int], int]
            weights = default_weights,  # type: Callable[[int], np.ndarray]
            seed = None  # type: Optional[int]
    ):
        # type: (...) -> None
        
        self.config_space = get_variable_features(model)
        self.hyperparameters, self.losses = get_evaluations(model, num)

        self.parzen_estimator_parameters = ParzenEstimatorParameters(
            consider_prior, prior_weight, consider_magic_clip, consider_endpoints, weights)
        self.prior_weight = prior_weight
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.weights = weights
        self.seed = seed

        self.rng = np.random.RandomState(seed)
        self.random_sampler = random.RandomSampler(seed=seed)

    def sample(self):
        # type: (BaseStorage, int, str, BaseDistribution) -> float

        n = len(self.losses)

        if n < self.n_startup_trials:
            pass
            #return randomsample

        for var_name in self.hyperparameters.keys():            
            lower_vals, upper_vals = self._split_observation_pairs(var_name)

            _dist = distribution_type(self.config_space, var_name)

            if _dist == "cat":
                self._sample_categorical(var_name, lower_vals, upper_vals)
            else:
                self._sample_numerical(_dist, var_name, lower_vals, upper_vals)

    def _split_observation_pairs(self, var_name):
        observation_pairs = [[hp, loss] for hp, loss in zip(self.hyperparameters[var_name], self.losses) if hp != None]
        config_vals, loss_vals = np.asarray([p[0] for p in observation_pairs]), np.asarray([p[1] for p in observation_pairs])
        
        n_lower = self.gamma(len(config_vals))
        loss_ascending = np.argsort(loss_vals)

        lower_vals = np.asarray(config_vals[loss_ascending[:n_lower]], dtype = float)
        upper_vals = np.asarray(config_vals[loss_ascending[n_lower:]], dtype = float)

        return lower_vals, upper_vals

    def _sample_numerical(self, _dist, var_name, lower_vals, upper_vals):
        # type: (...) -> float
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

        size = (self.n_ei_candidates, )

        parzen_estimator_lower = ParzenEstimator(samples = lower_vals, lower = lower_bound, upper = upper_bound, parameters = self.parzen_estimator_parameters)
        samples_lower = self._sample_from_gmm(parzen_estimator=parzen_estimator_lower, lower = lower_bound, upper = upper_bound, var_type = _dist, is_log = is_log, size = size)
        log_likelihoods_lower = self._gmm_log_pdf(samples=samples_lower, parzen_estimator=parzen_estimator_lower, lower = lower_bound, upper = upper_bound, var_type = _dist, is_log=is_log)

        parzen_estimator_upper = ParzenEstimator(samples = upper_vals, lower = lower_bound, upper = upper_bound, parameters = self.parzen_estimator_parameters)

        log_likelihoods_upper = self._gmm_log_pdf(samples=samples_lower, parzen_estimator=parzen_estimator_upper, lower = lower_bound, upper = upper_bound, var_type = _dist, is_log = is_log)

        return float(TPESampler._compare(samples = samples_lower, log_l = log_likelihoods_lower, log_g = log_likelihoods_upper))

    def _sample_categorical(self, var_name, lower_vals, upper_vals):

        choices = self.config_space._hyperparameters[var_name].choices
        lower_vals = [choices.index(val) for val in lower_vals]
        upper_vals = [choices.index(val) for val in upper_vals]

        n_choices = len(choices)
        size = (self.n_ei_candidates, )

        weights_lower = self.weights(len(lower_vals))
        counts_lower = np.bincount(lower_vals, minlength = n_choices, weights = weights_lower)
        weighted_lower = counts_lower + self.prior_weight
        weighted_lower /= weighted_lower.sum()
        samples_lower = self._sample_from_categorical_dist(weighted_lower, size = size)
        log_likelihoods_lower = TPESampler._categorical_log_pdf(samples_lower, weighted_lower)

        weights_upper = self.weights(len(upper_vals))
        counts_upper = np.bincount(upper_vals, minlength = n_choices, weights = weights_upper)
        weighted_upper = counts_upper + self.prior_weight
        weighted_upper /= weighted_upper.sum()
        log_likelihoods_upper = TPESampler._categorical_log_pdf(samples_lower, weighted_upper)

        return int(TPESampler._compare(samples=samples_lower, log_l = log_likelihoods_lower, log_g=log_likelihoods_upper))

    def _sample_from_gmm(self, parzen_estimator, lower, upper, var_type, size=(), is_log = False):
        # type: (...) -> np.ndarray

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
            draw = self.rng.normal(loc=mus[active], scale=sigmas[active])
            if lower <= draw < upper:
                samples = np.append(samples, draw)

        samples = np.reshape(samples, size)

        if is_log:
            samples = np.exp(samples)

        if var_type == "float":
            return samples
        elif var_type == "int":
            return np.round(samples)

    def _gmm_log_pdf(self, samples, parzen_estimator, lower, upper, var_type, is_log = False):
        # type: (...) -> np.ndarray

        weights = parzen_estimator.weights
        mus = parzen_estimator.mus
        sigmas = parzen_estimator.sigmas
        samples, weights, mus, sigmas = map(np.asarray, (samples, weights, mus, sigmas))
        
        if samples.size == 0:
            return np.asarray([], dtype=float)
        if weights.ndim != 1:
            raise ValueError("The 'weights' should be 2-dimension. "
                             "But weights.shape = {}".format(weights.shape))
        if mus.ndim != 1:
            raise ValueError("The 'mus' should be 2-dimension. "
                             "But mus.shape = {}".format(mus.shape))
        if sigmas.ndim != 1:
            raise ValueError("The 'sigmas' should be 2-dimension. "
                             "But sigmas.shape = {}".format(sigmas.shape))
        
        p_accept = np.sum(weights * (TPESampler._normal_cdf(upper, mus, sigmas) - TPESampler._normal_cdf(lower, mus, sigmas)))

        if var_type == "float":
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
            probabilities = np.zeros(samples.shape, dtype=float)
            cdf_func = TPESampler._log_normal_cdf if is_log else TPESampler._normal_cdf
            for w, mu, sigma in zip(weights, mus, sigmas):
                if is_log:
                    upper_bound = np.minimum(samples + 0.5, np.exp(upper))
                    lower_bound = np.maximum(samples - 0.5, np.exp(lower))
                    lower_bound = np.maximum(0, lower_bound)
                else:
                    upper_bound = np.minimum(samples + 0.5, upper)
                    lower_bound = np.maximum(samples - 0.5, lower)
                probabilities += w * (cdf_func(upper_bound, mu, sigma) - cdf_func(lower_bound, mu, sigma))
            return_val = np.log(probabilities + EPS) - np.log(p_accept + EPS)

        return_val.shape = samples.shape
        
        return return_val

    def _sample_from_categorical_dist(self, probabilities, size=()):
        # type: (Union[np.ndarray, np.ndarray], Tuple) -> Union[np.ndarray, np.ndarray]

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
        assert len(size)
        assert probabilities.ndim == 1

        n_draws = int(np.prod(size))
        sample = self.rng.multinomial(n=1, pvals=probabilities, size=int(n_draws))
        assert sample.shape == size + (probabilities.size, )
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
    def _compare(cls, samples, log_l, log_g):
        
        samples, log_l, log_g = map(np.asarray, (samples, log_l, log_g))
        
        score = log_l - log_g
        if samples.size != score.size:
            raise ValueError("The size of the 'samples' and that of the 'score' "
                             "should be same. "
                             "But (samples.size, score.size) = ({}, {})".format(
                                 samples.size, score.size))

        best = np.argmax(score)
        
        return samples[best]
        
    @classmethod
    def _logsum_rows(cls, x):

        x = np.asarray(x)
        return np.log(np.exp(x).sum(axis=1))
        
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
        if x < 0:
            raise ValueError("Negative argument is given to _lognormal_cdf. x: {}".format(x))
        numerator = np.log(np.maximum(x, EPS)) - mu
        denominator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = numerator / denominator
        return .5 + .5 * scipy.special.erf(z)

