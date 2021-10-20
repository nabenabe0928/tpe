from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

import ConfigSpace as CS

from optimizer.parzen_estimator.loglikelihoods import compute_config_loglikelihoods
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
    default_percentile_maker,
    default_weights
)
from util.utils import (
    get_random_sample,
    revert_eval_config,
    store_results
)


ParzenEstimatorType = Union[NumericalParzenEstimator, CategoricalParzenEstimator]
HPType = Union[CategoricalHPType, NumericalHPType]


class PercentileFuncMaker(Protocol):
    def __call__(self, **kwargs: Dict[str, Any]) -> Callable[[np.ndarray], int]:
        ...


class TreeStructuredParzenEstimator:
    def __init__(self, config_space: CS.ConfigurationSpace,
                 percentile_func: Callable[[np.ndarray], int],
                 weight_func: Callable[[int, int], np.ndarray],
                 n_ei_candidates: int, metric_name: str = 'loss',
                 min_bandwidth_factor: float = 1e-2, seed: Optional[int] = None):
        """
        Attributes:
            rng (np.random.RandomState): random state to maintain the reproducibility
            n_ei_candidates (int): The number of samplings to optimize the EI value
            config_space (CS.ConfigurationSpace): The searching space of the task
            hp_names (List[str]): The list of hyperparameter names
            metric_name (str): The name of the metric (or objective function value)
            observations (Dict[str, Any]): The storage of the observations
            sorted_observations (Dict[str, Any]): The storage of the observations sorted based on loss
            min_bandwidth_factor (float): The minimum bandwidth for numerical parameters
            is_categoricals (Dict[str, bool]): Whether the given hyperparameter is categorical
            is_ordinals (Dict[str, bool]): Whether the given hyperparameter is ordinal
            percentile_func (Callable[[np.ndarray], int]):
                The function that returns the number of a better group based on the total number of evaluations.
            weight_func (Callable[[int, int], np.ndarray]):
                The function that returns the coefficients of each kernel.
        """
        self._rng = np.random.RandomState(seed)
        self._n_ei_candidates = n_ei_candidates
        self._config_space = config_space
        self._hp_names = list(config_space._hyperparameters.keys())
        self._metric_name = metric_name
        self._n_lower = 0
        self._min_bandwidth_factor = min_bandwidth_factor

        self._observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._sorted_observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._observations[self.metric_name] = np.array([])
        self._sorted_observations[self.metric_name] = np.array([])

        self._weight_func = weight_func
        self._percentile_func = percentile_func

        self._is_categoricals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == 'CategoricalHyperparameter'
            for hp_name in self._hp_names
        }

        self._is_ordinals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == 'OrdinalHyperparameter'
            for hp_name in self._hp_names
        }

    def set_prior_observations(self, observations: Dict[str, np.ndarray]) -> None:
        if self.observations[self.metric_name].size != 0:
            raise ValueError('Prior observations must be set before the optimization.')

        self._observations = observations
        order = np.argsort(self.observations[self.metric_name])
        self._sorted_observations = {
            hp_name: observations[order]
            for hp_name, observations in self.observations.items()
        }

    def update_observations(self, eval_config: Dict[str, NumericType], loss: float) -> None:
        """
        Update the observations for the TPE construction

        Args:
            eval_config (Dict[str, NumericType]): The configuration to evaluate (after conversion)
            loss (float): The loss value as a result of the evaluation
        """
        sorted_losses, losses = self.sorted_observations[self.metric_name], self.observations[self.metric_name]
        insert_loc = np.searchsorted(sorted_losses, loss, side='right')
        self._observations[self.metric_name] = np.append(losses, loss)
        self._sorted_observations[self.metric_name] = np.insert(sorted_losses, insert_loc, loss)
        self._n_lower = self.percentile_func(self._sorted_observations[self.metric_name])

        observations, sorted_observations = self._observations, self._sorted_observations
        for hp_name in self.hp_names:
            is_categorical = self.is_categoricals[hp_name]
            config = self.config_space.get_hyperparameter(hp_name)
            config_type = config.__class__.__name__
            val = eval_config[hp_name]

            if is_categorical:
                observations[hp_name] = np.append(observations[hp_name], val)
                if sorted_observations[hp_name].size == 0:  # cannot cast str to float!
                    sorted_observations[hp_name] = np.array([val], dtype='U32')
                else:
                    sorted_observations[hp_name] = np.insert(sorted_observations[hp_name], insert_loc, val)
            else:
                dtype = config2type[config_type]
                observations[hp_name] = np.append(observations[hp_name], val).astype(dtype)
                sorted_observations[hp_name] = np.insert(sorted_observations[hp_name], insert_loc, val).astype(dtype)

    def get_config_candidates(self) -> List[np.ndarray]:
        """
        Since we compute the probability improvement of each objective independently,
        we need to sample the configurations in advance.

        Returns:
            config_cands (List[np.ndarray]): arrays of candidates in each dimension
        """
        config_cands = []
        n_lower = self.n_lower

        for hp_name in self.hp_names:
            lower_vals = self.sorted_observations[hp_name][:n_lower]
            empty = np.array([lower_vals[0]])

            is_categorical = self.is_categoricals[hp_name]
            pe_lower, _ = self._get_parzen_estimator(lower_vals=lower_vals, upper_vals=empty,
                                                     hp_name=hp_name, is_categorical=is_categorical)

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
        """
        is_categorical = self.is_categoricals[hp_name]
        sorted_observations = self.sorted_observations[hp_name]
        n_lower = self.n_lower

        # split observations
        lower_vals = sorted_observations[:n_lower]
        upper_vals = sorted_observations[n_lower:]

        pe_lower, pe_upper = self._get_parzen_estimator(lower_vals=lower_vals, upper_vals=upper_vals,
                                                        hp_name=hp_name, is_categorical=is_categorical)

        return pe_lower.basis_loglikelihood(samples), pe_upper.basis_loglikelihood(samples)

    def compute_config_loglikelihoods(self, config_cands: List[np.ndarray]
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the probability improvement given configurations

        Args:
            config_cands (List[np.ndarray]):
                The list of candidate values for each dimension.
                The length is the number of dimensions and
                each array has the length of n_ei_candidates.

        Returns:
            config_ll_lower, config_ll_upper (Tuple[np.ndarray]):
                The loglikelihoods of each configuration in
                the good group or bad group.
                The shape is (n_ei_candidates, ) for each.
        """
        dim = len(self.hp_names)
        n_evals = self.sorted_observations[self.metric_name].size
        n_lower = self.n_lower

        n_candidates = config_cands[0].size
        basis_loglikelihoods_lower = np.zeros((dim, n_lower + 1, n_candidates))
        weights_lower = self.weight_func(n_lower + 1)
        basis_loglikelihoods_upper = np.zeros((dim, n_evals - n_lower + 1, n_candidates))
        weights_upper = self.weight_func(n_evals - n_lower + 1)

        for dim, (hp_name, samples) in enumerate(zip(self.hp_names, config_cands)):
            bll_lower, bll_upper = self._compute_basis_loglikelihoods(hp_name=hp_name, samples=samples)
            basis_loglikelihoods_lower[dim] += bll_lower
            basis_loglikelihoods_upper[dim] += bll_upper

        config_ll_lower = compute_config_loglikelihoods(basis_loglikelihoods_lower, weights_lower)
        config_ll_upper = compute_config_loglikelihoods(basis_loglikelihoods_upper, weights_upper)

        return config_ll_lower, config_ll_upper

    def compute_probability_improvement(self, config_cands: List[np.ndarray]) -> np.ndarray:
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
        """
        config_ll_lower, config_ll_upper = self.compute_config_loglikelihoods(config_cands=config_cands)
        return config_ll_lower - config_ll_upper

    def _get_parzen_estimator(self, lower_vals: np.ndarray, upper_vals: np.ndarray, hp_name: str,
                              is_categorical: bool) -> Tuple[ParzenEstimatorType, ParzenEstimatorType]:
        """
        Construct parzen estimators for the lower and the upper groups and return them

        Args:
            lower_vals (np.ndarray): The array of the values in the lower group
            upper_vals (np.ndarray): The array of the values in the upper group
            hp_name (str): The name of the hyperparameter
            is_categorical (bool): Whether the given hyperparameter is categorical

        Returns:
            pe_lower (ParzenEstimatorType): The parzen estimator for the lower group
            pe_upper (ParzenEstimatorType): The parzen estimator for the upper group
        """
        config = self.config_space.get_hyperparameter(hp_name)
        config_type = config.__class__.__name__
        is_ordinal = self.is_ordinals[hp_name]
        parzen_estimator_args = dict(config=config, weight_func=self.weight_func)

        if is_categorical:
            pe_lower = build_categorical_parzen_estimator(vals=lower_vals, **parzen_estimator_args)
            pe_upper = build_categorical_parzen_estimator(vals=upper_vals, **parzen_estimator_args)
        else:
            min_bandwidth_factor = 1.0 / len(config.sequence) if is_ordinal else self.min_bandwidth_factor
            parzen_estimator_args.update(
                dtype=config2type[config_type],
                is_ordinal=is_ordinal,
                min_bandwidth_factor=min_bandwidth_factor
            )
            pe_lower = build_numerical_parzen_estimator(vals=lower_vals, **parzen_estimator_args)
            pe_upper = build_numerical_parzen_estimator(vals=upper_vals, **parzen_estimator_args)

        return pe_lower, pe_upper

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._config_space

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return self._observations

    @property
    def sorted_observations(self) -> Dict[str, np.ndarray]:
        return self._sorted_observations

    @property
    def is_categoricals(self) -> Dict[str, bool]:
        return self._is_categoricals

    @property
    def is_ordinals(self) -> Dict[str, bool]:
        return self._is_ordinals

    @property
    def rng(self) -> np.random.RandomState:
        return self._rng

    @property
    def n_ei_candidates(self) -> int:
        return self._n_ei_candidates

    @property
    def n_lower(self) -> int:
        return self._n_lower

    @property
    def hp_names(self) -> List[str]:
        return self._hp_names

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def min_bandwidth_factor(self) -> float:
        return self._min_bandwidth_factor

    @property
    def weight_func(self) -> Callable:
        return self._weight_func

    @property
    def percentile_func(self) -> Callable:
        return self._percentile_func


class TPEOptimizer:
    def __init__(self, obj_func: Callable, config_space: CS.ConfigurationSpace,
                 resultfile: str, n_init: int = 10, max_evals: int = 100,
                 seed: Optional[int] = None, metric_name: str = 'loss',
                 n_ei_candidates: int = 24,
                 percentile_func_maker: PercentileFuncMaker = default_percentile_maker,
                 weight_func: Callable[[int, int], np.ndarray] = default_weights):
        """
        Attributes:
            rng (np.random.RandomState): random state to maintain the reproducibility
            resultfile (str): The name of the result file to output in the end
            n_init (int): The number of random sampling before using TPE
            obj_func (Callable): The objective function
            hp_names (List[str]): The list of hyperparameter names
            observations (Dict[str, Any]): The storage of the observations
            config_space (CS.ConfigurationSpace): The searching space of the task
            is_categoricals (Dict[str, bool]): Whether the given hyperparameter is categorical
            is_ordinals (Dict[str, bool]): Whether the given hyperparameter is ordinal
        """

        self._rng = np.random.RandomState(seed)
        self._n_init, self._max_evals = n_init, max_evals
        self.resultfile = resultfile
        self._obj_func = obj_func
        self._hp_names = list(config_space._hyperparameters.keys())

        self._config_space = config_space
        self._is_categoricals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == 'CategoricalHyperparameter'
            for hp_name in self._hp_names
        }
        self._is_ordinals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == 'OrdinalHyperparameter'
            for hp_name in self._hp_names
        }

        self.tpe = TreeStructuredParzenEstimator(
            config_space=config_space,
            metric_name=metric_name,
            n_ei_candidates=n_ei_candidates,
            seed=seed,
            percentile_func=percentile_func_maker(),
            weight_func=weight_func
        )

    def optimize(self, logger: Logger) -> Tuple[Dict[str, Any], float]:
        """
        Optimize obj_func using TPE Sampler and store the results in the end.

        Args:
            logger (Logger): The logging to write the intermediate results

        Returns:
            best_config (Dict[str, Any]): The configuration that has the best loss
            best_loss (float): The best loss value during the optimization
        """

        best_config, best_loss, t = {}, np.inf, 0

        while True:
            logger.info(f'\nIteration: {t + 1}')
            eval_config = self.random_sample() if t < self.n_init else self.sample()

            loss = self.obj_func(eval_config)
            self.tpe.update_observations(eval_config=eval_config, loss=loss)

            if best_loss > loss:
                best_loss = loss
                best_config = eval_config

            logger.info('Cur. loss: {:.4f}, Cur. Config: {}'.format(loss, eval_config))
            logger.info('Best loss: {:.4f}, Best Config: {}'.format(best_loss, best_config))
            t += 1

            if t >= self.max_evals:
                break

        store_results(best_config=best_config, logger=logger,
                      observations=self.tpe.observations, file_name=self.resultfile)

        return best_config, best_loss

    def sample(self) -> Dict[str, Any]:
        """
        Sample a configuration using tree-structured parzen estimator (TPE)

        Returns:
            eval_config (Dict[str, Any]): A sampled configuration from TPE
        """
        config_cands = self.tpe.get_config_candidates()

        pi_config = self.tpe.compute_probability_improvement(config_cands=config_cands)
        best_idx = int(np.argmax(pi_config))
        eval_config = {hp_name: config_cands[dim][best_idx]
                       for dim, hp_name in enumerate(self.hp_names)}

        return self._revert_eval_config(eval_config=eval_config)

    def _get_random_sample(self, hp_name: str) -> NumericType:
        return get_random_sample(hp_name=hp_name, rng=self.rng, config_space=self.config_space,
                                 is_categorical=self.is_categoricals[hp_name],
                                 is_ordinal=self.is_ordinals[hp_name])

    def _revert_eval_config(self, eval_config: Dict[str, NumericType]) -> Dict[str, Any]:
        return revert_eval_config(eval_config=eval_config, config_space=self.config_space,
                                  is_categoricals=self.is_categoricals, is_ordinals=self.is_ordinals,
                                  hp_names=self.hp_names)

    def random_sample(self) -> Dict[str, Any]:
        eval_config = {hp_name: self._get_random_sample(hp_name=hp_name) for hp_name in self.hp_names}
        return self._revert_eval_config(eval_config=eval_config)

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._config_space

    @property
    def hp_names(self) -> List[str]:
        return self._hp_names

    @property
    def is_categoricals(self) -> Dict[str, bool]:
        return self._is_categoricals

    @property
    def is_ordinals(self) -> Dict[str, bool]:
        return self._is_ordinals

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
