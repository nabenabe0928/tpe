from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Type

import numpy as np

import ConfigSpace as CS

from tpe.optimizer.base_optimizer import BaseOptimizer
from tpe.optimizer.parzen_estimator.loglikelihoods import compute_config_loglikelihoods
from tpe.optimizer.parzen_estimator.parzen_estimator import (
    ParzenEstimatorType,
    build_categorical_parzen_estimator,
    build_numerical_parzen_estimator,
)
from tpe.utils.constants import (
    NumericType,
    config2type,
    default_percentile_maker,
    uniform_weight,
)


class PercentileFuncMaker(Protocol):
    def __call__(self, **kwargs: Dict[str, Any]) -> Callable[[np.ndarray], int]:
        raise NotImplementedError


class TreeStructuredParzenEstimator:
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        percentile_func: Callable[[np.ndarray], int],
        n_ei_candidates: int,
        metric_name: str = "loss",
        runtime_name: str = "iter_time",
        min_bandwidth_factor: float = 1e-2,
        seed: Optional[int] = None,
    ):
        """
        Attributes:
            rng (np.random.RandomState): random state to maintain the reproducibility
            n_ei_candidates (int): The number of samplings to optimize the EI value
            config_space (CS.ConfigurationSpace): The searching space of the task
            hp_names (List[str]): The list of hyperparameter names
            metric_name (str): The name of the metric (or objective function value)
            runtime_name (str): The name of the runtime metric.
            observations (Dict[str, Any]): The storage of the observations
            sorted_observations (Dict[str, Any]): The storage of the observations sorted based on loss
            min_bandwidth_factor (float): The minimum bandwidth for numerical parameters
            is_categoricals (Dict[str, bool]): Whether the given hyperparameter is categorical
            is_ordinals (Dict[str, bool]): Whether the given hyperparameter is ordinal
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
        self._min_bandwidth_factor = min_bandwidth_factor

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
        self._pe_lower_dict: Dict[str, ParzenEstimatorType] = {}
        self._pe_upper_dict: Dict[str, ParzenEstimatorType] = {}

    def _insert_observations(
        self, key: str, insert_loc: int, val: Any, is_categorical: bool, dtype: Optional[Type[np.number]] = None
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

    def update_observations(self, eval_config: Dict[str, NumericType], loss: float, runtime: float) -> None:
        """
        Update the observations for the TPE construction

        Args:
            eval_config (Dict[str, NumericType]): The configuration to evaluate (after conversion)
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

        for hp_name in self._hp_names:
            is_categorical = self._is_categoricals[hp_name]
            config_type = self._config_space.get_hyperparameter(hp_name).__class__.__name__

            self._insert_observations(
                key=hp_name,
                insert_loc=insert_loc,
                val=eval_config[hp_name],
                is_categorical=is_categorical,
                dtype=config2type[config_type] if not is_categorical else None
            )
            self._update_parzen_estimators(hp_name)
        else:
            runtime_key = self._runtime_name
            self._insert_observations(
                key=runtime_key,
                insert_loc=insert_loc,
                val=runtime,
                is_categorical=False,
                dtype=np.float32
            )

    def _update_parzen_estimators(self, hp_name: str) -> None:
        is_categorical = self._is_categoricals[hp_name]
        sorted_observations = self._sorted_observations[hp_name]
        n_lower = self._n_lower

        # split observations
        lower_vals = sorted_observations[:n_lower]
        upper_vals = sorted_observations[n_lower:]

        pe_lower, pe_upper = self._get_parzen_estimator(
            lower_vals=lower_vals,
            upper_vals=upper_vals,
            hp_name=hp_name,
            is_categorical=is_categorical,
        )
        self._pe_lower_dict[hp_name] = pe_lower
        self._pe_upper_dict[hp_name] = pe_upper

    def get_config_candidates(self) -> List[np.ndarray]:
        """
        Since we compute the probability improvement of each objective independently,
        we need to sample the configurations in advance.

        Returns:
            config_cands (List[np.ndarray]): arrays of candidates in each dimension
        """
        config_cands = [
            self._pe_lower_dict[hp_name].sample(self._rng, self._n_ei_candidates) for hp_name in self._hp_names
        ]
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
        pe_lower = self._pe_lower_dict[hp_name]
        pe_upper = self._pe_upper_dict[hp_name]
        bll_lower = pe_lower.basis_loglikelihood(samples)
        bll_upper = pe_upper.basis_loglikelihood(samples)
        return bll_lower, bll_upper

    def compute_config_loglikelihoods(self, config_cands: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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
        dim = len(self._hp_names)
        n_evals = self._sorted_observations[self._metric_name].size
        n_lower = self._n_lower

        n_candidates = config_cands[0].size
        blls_lower = np.zeros((dim, n_lower + 1, n_candidates))
        weights_lower = uniform_weight(n_lower + 1)
        blls_upper = np.zeros((dim, n_evals - n_lower + 1, n_candidates))
        weights_upper = uniform_weight(n_evals - n_lower + 1)

        for dim, (hp_name, samples) in enumerate(zip(self._hp_names, config_cands)):
            bll_lower, bll_upper = self._compute_basis_loglikelihoods(hp_name, samples)
            blls_lower[dim] += bll_lower
            blls_upper[dim] += bll_upper

        config_ll_lower = compute_config_loglikelihoods(blls_lower, weights_lower)
        config_ll_upper = compute_config_loglikelihoods(blls_upper, weights_upper)

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
        cll_lower, cll_upper = self.compute_config_loglikelihoods(config_cands)
        return cll_lower - cll_upper

    def _get_parzen_estimator(
        self,
        lower_vals: np.ndarray,
        upper_vals: np.ndarray,
        hp_name: str,
        is_categorical: bool,
    ) -> Tuple[ParzenEstimatorType, ParzenEstimatorType]:
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
            pe_lower = build_categorical_parzen_estimator(vals=lower_vals, **kwargs)
            pe_upper = build_categorical_parzen_estimator(vals=upper_vals, **kwargs)
        else:
            min_bandwidth_factor = 1.0 / len(config.sequence) if is_ordinal else self._min_bandwidth_factor
            kwargs.update(
                dtype=config2type[config_type],
                is_ordinal=is_ordinal,
                min_bandwidth_factor=min_bandwidth_factor,
            )
            pe_lower = build_numerical_parzen_estimator(vals=lower_vals, **kwargs)
            pe_upper = build_numerical_parzen_estimator(vals=upper_vals, **kwargs)

        return pe_lower, pe_upper

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return {hp_name: vals.copy() for hp_name, vals in self._observations.items()}


class TPEOptimizer(BaseOptimizer):
    def __init__(
        self,
        obj_func: Callable,
        config_space: CS.ConfigurationSpace,
        resultfile: str,
        n_init: int = 10,
        max_evals: int = 100,
        seed: Optional[int] = None,
        metric_name: str = "loss",
        runtime_name: str = "iter_time",
        only_requirements: bool = False,
        n_ei_candidates: int = 24,
        percentile_func_maker: PercentileFuncMaker = default_percentile_maker,
    ):

        super().__init__(
            obj_func=obj_func,
            config_space=config_space,
            resultfile=resultfile,
            n_init=n_init,
            max_evals=max_evals,
            seed=seed,
            metric_name=metric_name,
            runtime_name=runtime_name,
            only_requirements=only_requirements
        )

        self._tpe = TreeStructuredParzenEstimator(
            config_space=config_space,
            metric_name=metric_name,
            runtime_name=runtime_name,
            n_ei_candidates=n_ei_candidates,
            seed=seed,
            percentile_func=percentile_func_maker(),
        )

    def update(self, eval_config: Dict[str, Any], loss: float, runtime: float) -> None:
        self._tpe.update_observations(eval_config=eval_config, loss=loss, runtime=runtime)

    def fetch_observations(self) -> Dict[str, np.ndarray]:
        return self._tpe.observations

    def sample(self) -> Dict[str, Any]:
        """
        Sample a configuration using tree-structured parzen estimator (TPE)

        Returns:
            eval_config (Dict[str, Any]): A sampled configuration from TPE
        """
        config_cands = self._tpe.get_config_candidates()

        pi_config = self._tpe.compute_probability_improvement(config_cands=config_cands)
        best_idx = int(np.argmax(pi_config))
        eval_config = {hp_name: config_cands[dim][best_idx] for dim, hp_name in enumerate(self._hp_names)}

        return self._revert_eval_config(eval_config=eval_config)
