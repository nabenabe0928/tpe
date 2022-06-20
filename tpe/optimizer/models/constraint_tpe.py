from typing import Any, Dict, List, Optional, Union

import ConfigSpace as CS

import numpy as np

from tpe.optimizer.models import AbstractTPE, MultiObjectiveTPE, TPE
from tpe.utils.constants import NumericType


OBJECTIVE_KEY = "objective"
TPESamplerType = Union[TPE, MultiObjectiveTPE]


def _copy_observations(observations: Dict[str, np.ndarray], param_names: List[str]) -> Dict[str, np.ndarray]:
    return {param_name: observations[param_name].copy() for param_name in param_names}


class ConstraintTPE(AbstractTPE):
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        n_ei_candidates: int,
        objective_names: List[str],
        runtime_name: str,
        seed: Optional[int],
        min_bandwidth_factor: float,
        top: float,
        constraints: Dict[str, float],
    ):
        tpe_params = dict(
            config_space=config_space,
            n_ei_candidates=n_ei_candidates,
            runtime_name=runtime_name,
            seed=seed,
            min_bandwidth_factor=min_bandwidth_factor,
            top=top,
        )
        self._constraints = constraints.copy()
        self._objective_names = objective_names[:]
        self._samplers: Dict[str, TPESamplerType] = {}
        self._satisfied_flag = np.array([])
        self._feasible_counts = {name: 0 for name in constraints.keys()}
        self._init_samplers(objective_names, tpe_params)

    def _init_samplers(self, objective_names: List[str], tpe_params: Dict[str, Any]) -> None:
        if len(objective_names) == 1:
            self._samplers[OBJECTIVE_KEY] = TPE(objective_name=objective_names[0], **tpe_params)
        else:
            self._samplers[OBJECTIVE_KEY] = MultiObjectiveTPE(objective_names=objective_names, **tpe_params)

        for name, threshold in self._constraints.items():
            self._samplers[name] = TPE(objective_name=name, **tpe_params)

    def _is_satisfied(self, results: Dict[str, float]) -> bool:
        return all(results[name] <= threshold for name, threshold in self._constraints.items())

    def _validate_observations(self, n_observations: int, observations: Dict[str, np.ndarray]) -> None:
        for vals in observations.values():
            if vals.size != n_observations:
                raise ValueError("Each parameter must have the same number of observations")

    def _get_is_satisfied_flag_from_data(self, n_observations: int, observations: Dict[str, np.ndarray]) -> np.ndarray:
        satisfied_flag = np.zeros(n_observations, dtype=np.bool8)
        for idx in range(n_observations):
            results = {name: observations[name][idx] for name in self._constraints.keys()}
            satisfied_flag[idx] = self._is_satisfied(results)
        return satisfied_flag

    def apply_knowledge_augmentation(self, observations: Dict[str, np.ndarray]) -> None:
        main_sampler = self._samplers[OBJECTIVE_KEY]
        n_observations = len(list(observations.values()))
        hp_names = main_sampler._hp_names[:]
        self._validate_observations(n_observations=n_observations, observations=observations)
        if any(main_sampler._observations[objective_name].size != 0 for objective_name in self._objective_names):
            raise ValueError("Knowledge augmentation must be applied before the optimization.")

        all_objectives_exist = all(observations[objective_name].size != 0 for objective_name in self._objective_names)
        all_constraints_exist = all(observations[name].size != 0 for name in self._constraints.keys())
        if all_objectives_exist and all_constraints_exist:
            self._satisfied_flag = self._get_is_satisfied_flag_from_data(n_observations, observations)

        if all_objectives_exist:
            _observations = _copy_observations(observations=observations, param_names=hp_names + self._objective_names)
            main_sampler.apply_knowledge_augmentation(observations=_observations)

        for name, threshold in self._constraints.items():
            if name not in observations:
                continue

            _observations = _copy_observations(observations=observations, param_names=hp_names + [name])
            self._feasible_counts[name] = np.sum(observations[name] <= threshold)
            self._samplers[name].apply_knowledge_augmentation(observations=_observations)

    def _percentile_func_for_objective(self) -> int:
        sampler = self._samplers[OBJECTIVE_KEY]
        n_lower = sampler._percentile_func()
        sorted_satisfied_flag = self._satisfied_flag[sampler._order]
        n_satisfied = sorted_satisfied_flag.cumsum()
        # Take at least `n_lower` feasible solutions in the better group
        idx = np.searchsorted(n_satisfied, n_lower, side="left") + 1
        return min(idx, n_satisfied.size)

    def update_observations(
        self, eval_config: Dict[str, NumericType], results: Dict[str, float], runtime: float
    ) -> None:
        """
        Update the observations for the TPE construction

        Args:
            eval_config (Dict[str, NumericType]): The configuration to evaluate (after conversion)
            results (Dict[str, float]): The dict of loss values.
            runtime (float): The runtime for both sampling and training
        """
        self._satisfied_flag = np.append(self._satisfied_flag, self._is_satisfied(results))
        _results = {objective_name: results[objective_name] for objective_name in self._objective_names}
        self._samplers[OBJECTIVE_KEY].update_observations(
            results=_results,
            eval_config=eval_config,
            runtime=runtime,
            percentile_func=self._percentile_func_for_objective,
        )

        for name, threshold in self._constraints.items():
            self._feasible_counts[name] += results[name] <= threshold
            self._samplers[name].update_observations(
                results={name: results[name]},
                eval_config=eval_config,
                runtime=runtime,
                percentile_func=lambda: max(1, self._feasible_counts[name]),
            )

    def get_config_candidates(self) -> Dict[str, np.ndarray]:
        """
        Since we compute the probability improvement of each objective independently,
        we need to sample the configurations in advance.

        Returns:
            config_cands (Dict[str, np.ndarray]):
                A dict of arrays of candidates in each dimension
        """
        config_cands: Dict[str, np.ndarray] = {}
        for sampler in self._samplers.values():
            configs = sampler.get_config_candidates()
            if len(config_cands) == 0:
                config_cands = configs
            else:
                config_cands = {
                    hp_name: np.concatenate([config_cands[hp_name], configs[hp_name]]) for hp_name in configs.keys()
                }

        return config_cands

    def compute_probability_improvement(self, config_cands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute the (log) probability improvement given configurations

        Args:
            config_cands (Dict[str, np.ndarray]):
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
        pi_config = self._samplers[OBJECTIVE_KEY].compute_probability_improvement(config_cands)
        for name in self._constraints.keys():
            pi_config += self._samplers[name].compute_probability_improvement(config_cands)

        return pi_config

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        observations = self._samplers[OBJECTIVE_KEY].observations
        n_evals = observations[self._objective_names[0]].size
        for name in self._constraints.keys():
            # Some constraints might be augmented and thus we need to take the latest n_evals values.
            observations[name] = self._samplers[name]._observations[name][-n_evals:]

        return observations
