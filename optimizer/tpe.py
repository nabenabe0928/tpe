from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import json

import ConfigSpace as CS

from optimizer.parzen_estimator.loglikelihoods import compute_config_loglikelihoods_ratio
from optimizer.parzen_estimator.parzen_estimator import (
    CategoricalParzenEstimator,
    NumericalParzenEstimator,
    build_categorical_parzen_estimator,
    build_numerical_parzen_estimator
)
from util.constants import (
    CategoricalHPType,
    NumericType,
    NumericalHPType,
    config2type,
    default_percentile,
    default_weights
)
from util.utils import get_random_sample, revert_eval_config, save_observations


ParzenEstimatorType = Union[NumericalParzenEstimator, CategoricalParzenEstimator]
HPType = Union[CategoricalHPType, NumericalHPType]


class TPE:
    def __init__(self, obj_func: Callable, config_space: CS.ConfigurationSpace,
                 resultfile: str, mutation_prob: float = 0.05, n_init: int = 10,
                 max_evals: int = 100, seed: Optional[int] = None,
                 n_ei_candidates: int = 24, percentile_func: Callable = default_percentile,
                 weight_func: Callable = default_weights):
        """
        Attributes:
            rng (np.random.RandomState): random state to maintain the reproducibility
            resultfile (str): The name of the result file to output in the end
            n_ei_candidates (int): The number of samplings to optimize the EI value
            n_init (int): The number of random sampling before using TPE
            obj_func (Callable): The objective function
            hp_names (List[str]): The list of hyperparameter names
            mutation_prob (float): The probablity to change a parameter to a different value
            observations (Dict[str, Any]): The storage of the observations
            config_space (CS.ConfigurationSpace): The searching space of the task
            percentile_func (Callable):
                The function that returns the number of a better group based on the total number of evaluations.
            weight_func: callable
                The function that returns the coefficients of each kernel.
        """

        self._rng = np.random.RandomState(seed)
        self._n_init, self._max_evals = n_init, max_evals
        self.resultfile = resultfile
        self._n_ei_candidates = n_ei_candidates
        self._obj_func = obj_func
        self._mutation_prob = mutation_prob
        self._hp_names = list(config_space._hyperparameters.keys())

        self._observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._order = np.array([])

        self._config_space = config_space
        self._percentile_func = percentile_func
        self._weight_func = weight_func
        self._is_categorical = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == 'CategoricalHyperparameter'
            for hp_name in self._hp_names
        }

    def _update_observations(self, eval_config: Dict[str, NumericType], loss: float) -> None:
        """
        Update the observations for the TPE construction

        Args:
            eval_config (Dict[str, NumericType]): The configuration to evaluate (after conversion)
            loss (float): The loss value as a result of the evaluation

        TODO: Test the change by the update
        """
        self._observations['loss'] = np.append(self._observations['loss'], loss)
        self._order = np.argsort(self.observations['loss'])

        observations = self._observations
        for hp_name in self.hp_names:
            is_categorical = self.is_categorical[hp_name]
            config = self.config_space.get_hyperparameter(hp_name)
            config_type = config.__class__.__name__
            val = eval_config[hp_name]

            if is_categorical:
                observations[hp_name] = np.append(observations[hp_name], val)
            else:
                dtype = config2type[config_type]
                observations[hp_name] = np.append(observations[hp_name], val).astype(dtype)

    def _store_results(self, best_config: Dict[str, np.ndarray], logger: Logger) -> None:
        logger.info('\nThe observations: {}'.format(self.observations))
        save_observations(filename=self.resultfile, observations=self.observations)

        with open('opt_cfg.json', mode='w') as f:
            json.dump(best_config, f, indent=4)

    def optimize(self, logger: Logger) -> None:
        """
        Optimize obj_func using TPE Sampler and store the results in the end.

        Args:
            logger (Logger): The logging to write the intermediate results

        TODO: Add test by benchmark functions
        """
        self._observations['loss'] = np.array([])

        best_config, best_loss, t = {}, np.inf, 0

        while True:
            logger.info(f'\nIteration: {t + 1}')
            eval_config = self.random_sample() if t < self.n_init else self.sample()

            loss = self.obj_func(eval_config)
            self._update_observations(eval_config=eval_config, loss=loss)

            if best_loss > loss:
                best_loss = loss
                best_config = eval_config

            logger.info('Cur. loss: {:.4f}, Cur. Config: {}'.format(loss, eval_config))
            logger.info('Best loss: {:.4f}, Best Config: {}'.format(best_loss, best_config))
            t += 1

            if t >= self.max_evals:
                break

        self._store_results(best_config=best_config, logger=logger)

    def _get_config_candidates(self) -> List[np.ndarray]:
        """
        Since we compute the probability improvement of each objective independently,
        we need to sample the configurations in advance.

        Returns:
            (np.ndarray): An array of candidates in one dimension

        TODO: Test
        """
        config_cands = []
        n_evals = len(self.order)
        n_lowers = self.percentile_func(n_evals)

        for dim, hp_name in enumerate(self.hp_names):
            lower_vals = self.observations[hp_name][self.order[:n_lowers]]
            empty = np.array([lower_vals[0]])

            is_categorical = self.is_categorical[hp_name]
            config = self.config_space.get_hyperparameter(hp_name)
            pe_lower, _ = self._get_parzen_estimator(lower_vals=lower_vals, upper_vals=empty, config=config,
                                                     is_categorical=is_categorical)

            config_cands.append(pe_lower.sample(self.rng, self.n_ei_candidates))

        return config_cands

    def _compute_basis_loglikelihoods(self, hp_name: str, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the log likelihood of each basis of the provided hyperparameter

        Args:
            hp_name (str): The name of a hyperparameter
            samples (np.ndarray): The samples to compute the basis loglikelihoods

        Returns:
            basis_loglikelihoods (np.ndarray):
                The shape is (n_basis, n_samples).

        TODO: Add test
        """
        is_categorical = self.is_categorical[hp_name]
        ordered_observations = self.observations[hp_name][self.order]
        n_evals = len(self.order)
        n_lower = self.percentile_func(n_evals)

        # split observations
        lower_vals = ordered_observations[:n_lower]
        upper_vals = ordered_observations[n_lower:]

        config = self.config_space.get_hyperparameter(hp_name)
        pe_lower, pe_upper = self._get_parzen_estimator(lower_vals=lower_vals, upper_vals=upper_vals,
                                                        config=config, is_categorical=is_categorical)

        return pe_lower.basis_loglikelihood(samples), pe_upper.basis_loglikelihood(samples)

    def _compute_probability_improvement(self, config_cands: List[np.ndarray]) -> np.ndarray:
        """
        Compute the probability improvement given configurations

        Args:
            config_cands (List[np.ndarray]):
                The list of candidate values for each dimension.
                The length is the number of dimensions and
                each array has the length of n_ei_candidates.

        Returns:
            config_ll_ratio (np.ndarray):
                The log of the likelihood ratios of each configuration.
                The shape is (n_ei_candidates, )

        TODO: Add test
        """
        dim = len(self.hp_names)
        n_evals = len(self.order)
        n_lower = self.percentile_func(n_evals)

        basis_loglikelihoods_lower = np.zeros((dim, n_lower + 1, self.n_ei_candidates))
        weights_lower = self.weight_func(n_lower + 1)
        basis_loglikelihoods_upper = np.zeros((dim, n_evals - n_lower + 1, self.n_ei_candidates))
        weights_upper = self.weight_func(n_evals - n_lower + 1)

        for dim, (hp_name, samples) in enumerate(zip(self.hp_names, config_cands)):
            bll_lower, bll_upper = self._compute_basis_loglikelihoods(hp_name=hp_name, samples=samples)
            basis_loglikelihoods_lower[dim] += bll_lower
            basis_loglikelihoods_upper[dim] += bll_upper

        return compute_config_loglikelihoods_ratio(
            basis_loglikelihoods_lower=basis_loglikelihoods_lower,
            basis_loglikelihoods_upper=basis_loglikelihoods_upper,
            weights_lower=weights_lower,
            weights_upper=weights_upper
        )

    def sample(self) -> Dict[str, Any]:
        """
        Sample a configuration using tree-structured parzen estimator (TPE)

        Returns:
            eval_config (Dict[str, Any]): A sampled configuration from TPE
        """
        config_cands = self._get_config_candidates()

        pi_config = self._compute_probability_improvement(config_cands=config_cands)
        best_idx = int(np.argmax(pi_config))
        eval_config = {hp_name: self._get_random_sample(hp_name=hp_name)
                       if self.rng.random() < self.mutation_prob
                       else config_cands[dim][best_idx]
                       for dim, hp_name in enumerate(self.hp_names)}

        return self._revert_eval_config(eval_config=eval_config)

    def _get_parzen_estimator(self, lower_vals: np.ndarray, upper_vals: np.ndarray, config: HPType,
                              is_categorical: bool) -> Tuple[ParzenEstimatorType, ParzenEstimatorType]:
        """
        Construct parzen estimators for the lower and the upper groups and return them

        Args:
            lower_vals (np.ndarray): The array of the values in the lower group
            upper_vals (np.ndarray): The array of the values in the upper group
            config (HPType): The hyperparameter information (ConfigSpace)
            is_categorical (bool): Whether the given hyperparameter is categorical

        Returns:
            pe_lower (ParzenEstimatorType): The parzen estimator for the lower group
            pe_upper (ParzenEstimatorType): The parzen estimator for the upper group
        """
        config_type = config.__class__.__name__
        parzen_estimator_args = dict(config=config, weight_func=self.weight_func)

        if is_categorical:
            pe_lower = build_categorical_parzen_estimator(vals=lower_vals, **parzen_estimator_args)
            pe_upper = build_categorical_parzen_estimator(vals=upper_vals, **parzen_estimator_args)
        else:
            parzen_estimator_args.update(dtype=config2type[config_type])
            pe_lower = build_numerical_parzen_estimator(vals=lower_vals, **parzen_estimator_args)
            pe_upper = build_numerical_parzen_estimator(vals=upper_vals, **parzen_estimator_args)

        return pe_lower, pe_upper

    def _get_random_sample(self, hp_name: str) -> NumericType:
        return get_random_sample(hp_name=hp_name, rng=self.rng, config_space=self.config_space,
                                 is_categorical=self.is_categorical[hp_name])

    def _revert_eval_config(self, eval_config: Dict[str, Any]) -> Dict[str, Any]:
        return revert_eval_config(eval_config=eval_config, config_space=self.config_space,
                                  is_categoricals=self.is_categorical, hp_names=self.hp_names)

    def random_sample(self) -> Dict[str, Any]:
        eval_config = {hp_name: self._get_random_sample(hp_name=hp_name) for hp_name in self.hp_names}
        return self._revert_eval_config(eval_config=eval_config)

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._config_space

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return self._observations

    @property
    def hp_names(self) -> List[str]:
        return self._hp_names

    @property
    def is_categorical(self) -> Dict[str, bool]:
        return self._is_categorical

    @property
    def rng(self) -> np.random.RandomState:
        return self._rng

    @property
    def max_evals(self) -> int:
        return self._max_evals

    @property
    def n_init(self) -> int:
        return self._n_init

    @property
    def obj_func(self) -> Callable:
        return self._obj_func

    @property
    def order(self) -> np.ndarray:
        return self._order

    @property
    def percentile_func(self) -> Callable:
        return self._percentile_func

    @property
    def weight_func(self) -> Callable:
        return self._weight_func

    @property
    def mutation_prob(self) -> float:
        return self._mutation_prob

    @property
    def n_ei_candidates(self) -> int:
        return self._n_ei_candidates
