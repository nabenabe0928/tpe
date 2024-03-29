from __future__ import annotations

from collections.abc import Callable
from typing import Any, Type

import ConfigSpace as CS

import numpy as np

from parzen_estimator import (
    MultiVariateParzenEstimator,
    ParzenEstimatorType,
    build_categorical_parzen_estimator,
    build_numerical_parzen_estimator,
)

from tpe.utils.constants import NumericType, config2type


class TreeStructuredParzenEstimator:
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        percentile_func: Callable[[np.ndarray], int],
        n_ei_candidates: int,
        metric_name: str,
        runtime_name: str,
        seed: int | None,
        min_bandwidth_factor: float,
        top: float,
    ):
        """
        Attributes:
            rng (np.random.RandomState): random state to maintain the reproducibility
            n_ei_candidates (int): The number of samplings to optimize the EI value
            config_space (CS.ConfigurationSpace): The searching space of the task
            hp_names (List[str]): The list of hyperparameter names
            metric_name (str): The name of the metric (or objective function value)
            runtime_name (str): The name of the runtime metric.
            observations (dict[str, Any]): The storage of the observations
            sorted_observations (dict[str, Any]): The storage of the observations sorted based on loss
            min_bandwidth_factor (float): The minimum bandwidth for numerical parameters
            top (float): The hyperparam of the cateogircal kernel. It defines the prob of the top category.
            is_categoricals (dict[str, bool]): Whether the given hyperparameter is categorical
            is_ordinals (dict[str, bool]): Whether the given hyperparameter is ordinal
            percentile_func (Callable[[np.ndarray], int]):
                The function that returns the number of a better group based on the total number of evaluations.
        """
        self._rng = np.random.RandomState(seed)
        self._n_ei_candidates = n_ei_candidates
        self._config_space = config_space
        self._hp_names = list(config_space._hyperparameters.keys())
        self._metric_name = metric_name
        self._runtime_name = runtime_name
        self._n_lower = 0
        self._percentile = 0
        self._min_bandwidth_factor = min_bandwidth_factor
        self._top = top

        self._observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._sorted_observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._observations[self._metric_name] = np.array([])
        self._sorted_observations[self._metric_name] = np.array([])
        self._observations[self._runtime_name] = np.array([])
        self._sorted_observations[self._runtime_name] = np.array([])

        self._percentile_func = percentile_func

        self._is_categoricals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == "CategoricalHyperparameter"
            for hp_name in self._hp_names
        }

        self._is_ordinals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == "OrdinalHyperparameter"
            for hp_name in self._hp_names
        }
        self._mvpe_lower: MultiVariateParzenEstimator
        self._mvpe_upper: MultiVariateParzenEstimator

    def apply_knowledge_augmentation(self, observations: dict[str, np.ndarray]) -> None:
        if self.observations[self._metric_name].size != 0:
            raise ValueError("Knowledge augmentation must be applied before the optimization.")

        self._observations = {hp_name: vals.copy() for hp_name, vals in observations.items()}
        order = np.argsort(self.observations[self._metric_name])
        self._sorted_observations = {
            hp_name: observations[order] for hp_name, observations in self.observations.items()
        }
        self._n_lower = self._percentile_func(self._sorted_observations[self._metric_name])
        n_observations = self._observations[self._metric_name].size
        self._percentile = self._n_lower / n_observations
        self._update_parzen_estimators()

    def _insert_observations(
        self, key: str, insert_loc: int, val: Any, is_categorical: bool, dtype: Type[np.number] | None = None
    ) -> None:
        data, sorted_data = self._observations, self._sorted_observations

        if is_categorical:
            data[key] = np.append(data[key], val)
            if sorted_data[key].size == 0:  # cannot cast str to float!
                sorted_data[key] = np.array([val], dtype="U32")
            else:
                sorted_data[key] = np.insert(sorted_data[key], insert_loc, val)
        else:
            if dtype is None:
                raise ValueError("dtype must be specified, but got None")

            data[key] = np.append(data[key], val).astype(dtype)
            sorted_data[key] = np.insert(sorted_data[key], insert_loc, val).astype(dtype)

    def update_observations(self, eval_config: dict[str, NumericType], loss: float, runtime: float) -> None:
        """
        Update the observations for the TPE construction.
        Users can customize here by inheriting the class.

        Args:
            eval_config (dict[str, NumericType]): The configuration to evaluate (after conversion)
            loss (float): The loss value as a result of the evaluation
            runtime (float): The runtime for both sampling and training
        """
        self._update_observations(eval_config, loss, runtime)

    def _update_observations(self, eval_config: dict[str, NumericType], loss: float, runtime: float) -> None:
        """
        Update the observations for the TPE construction

        Args:
            eval_config (dict[str, NumericType]): The configuration to evaluate (after conversion)
            loss (float): The loss value as a result of the evaluation
            runtime (float): The runtime for both sampling and training
        """
        sorted_losses, losses = (
            self._sorted_observations[self._metric_name],
            self._observations[self._metric_name],
        )
        insert_loc = np.searchsorted(sorted_losses, loss, side="right")
        self._observations[self._metric_name] = np.append(losses, loss)
        self._sorted_observations[self._metric_name] = np.insert(sorted_losses, insert_loc, loss)
        self._n_lower = self._percentile_func(self._sorted_observations[self._metric_name])
        self._percentile = self._n_lower / self._observations[self._metric_name].size

        for hp_name in self._hp_names:
            is_categorical = self._is_categoricals[hp_name]
            config_type = self._config_space.get_hyperparameter(hp_name).__class__.__name__
            self._insert_observations(
                key=hp_name,
                insert_loc=insert_loc,
                val=eval_config[hp_name],
                is_categorical=is_categorical,
                dtype=config2type[config_type] if not is_categorical else None,
            )
        else:
            self._update_parzen_estimators()
            runtime_key = self._runtime_name
            self._insert_observations(
                key=runtime_key, insert_loc=insert_loc, val=runtime, is_categorical=False, dtype=np.float32
            )

    def _update_parzen_estimators(self) -> None:
        n_lower = self._n_lower
        pe_lower_dict: dict[str, ParzenEstimatorType] = {}
        pe_upper_dict: dict[str, ParzenEstimatorType] = {}
        for hp_name in self._hp_names:
            is_categorical = self._is_categoricals[hp_name]
            sorted_observations = self._sorted_observations[hp_name]

            # split observations
            lower_vals = sorted_observations[:n_lower]
            upper_vals = sorted_observations[n_lower:]

            pe_lower_dict[hp_name], pe_upper_dict[hp_name] = self._get_parzen_estimator(
                lower_vals=lower_vals,
                upper_vals=upper_vals,
                hp_name=hp_name,
                is_categorical=is_categorical,
            )

        self._mvpe_lower = MultiVariateParzenEstimator(pe_lower_dict)
        self._mvpe_upper = MultiVariateParzenEstimator(pe_upper_dict)

    def get_config_candidates(self, n_samples: int | None = None) -> dict[str, np.ndarray]:
        """
        Since we compute the probability improvement of each objective independently,
        we need to sample the configurations in advance.

        Args:
            n_samples (int):
                The number of samples.
                If None is provided, we use n_ei_candidates.

        Returns:
            config_cands (dict[str, np.ndarray]):
                A dict of arrays of candidates in each dimension
        """
        n_samples = n_samples if n_samples is not None else self._n_ei_candidates
        return self._mvpe_lower.sample(n_samples=n_samples, rng=self._rng, dim_independent=True, return_dict=True)

    def compute_config_loglikelihoods(self, config_cands: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the probability improvement given configurations

        Args:
            config_cands (dict[str, np.ndarray]):
                The dict of candidate values for each dimension.
                The length is the number of dimensions and
                each array has the length of n_ei_candidates.

        Returns:
            config_ll_lower, config_ll_upper (tuple[np.ndarray]):
                The loglikelihoods of each configuration in
                the good group or bad group.
                The shape is (n_ei_candidates, ) for each.
        """
        config_ll_lower = self._mvpe_lower.log_pdf(config_cands)
        config_ll_upper = self._mvpe_upper.log_pdf(config_cands)
        return config_ll_lower, config_ll_upper

    def compute_probability_improvement(self, config_cands: dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute the probability improvement given configurations

        Args:
            config_cands (dict[str, np.ndarray]):
                The dict of candidate values for each dimension.
                The length is the number of dimensions and
                each array has the length of n_ei_candidates.

        Returns:
            config_ll_ratio (np.ndarray):
                The log of the likelihood ratios of each configuration.
                The shape is (n_ei_candidates, )

        Note:
            In this implementation, we consider the gamma
                (gamma + (1 - gamma)g(x)/l(x))^-1
                = exp(log(gamma)) + exp(log(1 - gamma) + log(g(x)/l(x)))
        """
        EPS = 1e-12
        cll_lower, cll_upper = self.compute_config_loglikelihoods(config_cands)
        first_term = np.log(self._percentile + EPS)
        second_term = np.log(1.0 - self._percentile + EPS) + cll_upper - cll_lower
        pi = -np.logaddexp(first_term, second_term)
        return pi

    def _get_parzen_estimator(
        self,
        lower_vals: np.ndarray,
        upper_vals: np.ndarray,
        hp_name: str,
        is_categorical: bool,
    ) -> tuple[ParzenEstimatorType, ParzenEstimatorType]:
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
        pe_lower: ParzenEstimatorType
        pe_upper: ParzenEstimatorType

        config = self._config_space.get_hyperparameter(hp_name)
        config_type = config.__class__.__name__
        is_ordinal = self._is_ordinals[hp_name]
        kwargs = dict(config=config)

        if is_categorical:
            pe_lower = build_categorical_parzen_estimator(vals=lower_vals, **kwargs, top=self._top)
            pe_upper = build_categorical_parzen_estimator(vals=upper_vals, **kwargs, top=self._top)
        else:
            kwargs.update(
                dtype=config2type[config_type],
                is_ordinal=is_ordinal,
                default_min_bandwidth_factor=self._min_bandwidth_factor,
            )
            pe_lower = build_numerical_parzen_estimator(vals=lower_vals, **kwargs)
            pe_upper = build_numerical_parzen_estimator(vals=upper_vals, **kwargs)

        return pe_lower, pe_upper

    @property
    def observations(self) -> dict[str, np.ndarray]:
        return {hp_name: vals.copy() for hp_name, vals in self._observations.items()}
