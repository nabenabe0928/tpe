from typing import Dict, List, Optional, Union

import ConfigSpace as CS

import numpy as np

from tpe.optimizer.models import AbstractTPE, MultiObjectiveTPE, TPE
from tpe.utils.constants import NumericType


OBJECTIVE_KEY = "objective"


def _copy_observations(observations: Dict[str, np.ndarray], param_names: List[str]) -> Dict[str, np.ndarray]:
    return {param_name: observations[param_name].copy() for param_name in param_names}


class ConstraintTPE(AbstractTPE):
    # TODO: Make it possible to input percentile func from outside as used to be
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        n_ei_candidates: int,
        metric_names: List[str],
        constraints: Dict[str, float],
        runtime_name: str,
        seed: Optional[int],
        min_bandwidth_factor: float,
        top: float,
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
        self._metric_names = metric_names[:]
        self._samplers: Dict[str, Union[TPE, MultiObjectiveTPE]] = {}
        self._satisfied_flag = np.array([])
        if len(metric_names) == 1:
            self._samplers[OBJECTIVE_KEY] = TPE(metric_name=metric_names[0], **tpe_params)
        else:
            self._samplers[OBJECTIVE_KEY] = MultiObjectiveTPE(metric_names=metric_names, **tpe_params)

    def _is_satisfied(self, results: Dict[str, float]) -> bool:
        return all(results[metric_name] <= threshold for metric_name, threshold in self._constraints.items())

    def apply_knowledge_augmentation(self, observations: Dict[str, np.ndarray]) -> None:
        main_sampler = self._samplers[OBJECTIVE_KEY]
        if any(main_sampler._observations[metric_name].size != 0 for metric_name in self._metric_names):
            raise ValueError("Knowledge augmentation must be applied before the optimization.")

        hp_names = main_sampler._hp_names
        if all(observations[metric_name].size != 0 for metric_name in self._metric_names):
            main_sampler.apply_knowledge_augmentation(
                observations=_copy_observations(observations=observations, param_names=hp_names + self._metric_names)
            )

        for metric_name in self._constraints.keys():
            self._samplers[metric_name].apply_knowledge_augmentation(
                observations=_copy_observations(observations=observations, param_names=hp_names + [metric_name])
            )

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
        self._samplers[OBJECTIVE_KEY].update_observations(
            eval_config=eval_config,
            results={metric_name: results[metric_name] for metric_name in self._metric_names},
            runtime=runtime,
        )
        for metric_name in self._constraints.keys():
            self._samplers[metric_name].update_observations(
                eval_config=eval_config,
                results={metric_name: results[metric_name]},
                runtime=runtime,
            )
        self._satisfied_flag = np.append(self._satisfied_flag, self._is_satisfied(results))

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
        for metric_name in self._constraints.keys():
            pi_config += self._samplers[metric_name].compute_probability_improvement(config_cands)

        return pi_config

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        observations = self._samplers[OBJECTIVE_KEY].observations
        for metric_name in self._constraints.keys():
            observations[metric_name] = self._samplers[metric_name]._observations[metric_name].copy()

        return observations
