from __future__ import annotations

from typing import Any

import ConfigSpace as CS

import numpy as np

from tpe.optimizer.base_optimizer import BaseOptimizer, ObjectiveFunc
from tpe.optimizer.tpe import TreeStructuredParzenEstimator
from tpe.utils.constants import PercentileFuncMaker, default_percentile_maker


class TPEOptimizer(BaseOptimizer):
    def __init__(
        self,
        obj_func: ObjectiveFunc,
        config_space: CS.ConfigurationSpace,
        resultfile: str = "temp",
        n_init: int = 10,
        max_evals: int = 100,
        seed: int | None = None,
        metric_name: str = "loss",
        runtime_name: str = "iter_time",
        only_requirements: bool = False,
        n_ei_candidates: int = 24,
        # TODO: task names for transfer learning
        result_keys: list[str] | None = None,
        min_bandwidth_factor: float = 1e-1,
        top: float = 1.0,
        # TODO: Make dict of percentile_func_maker
        percentile_func_maker: PercentileFuncMaker = default_percentile_maker,  # type: ignore[assignment]
    ):
        """
        Args:
            obj_func (ObjectiveFunc): The objective function.
            config_space (CS.ConfigurationSpace): The searching space of the task
            resultfile (str): The name of the result file to output in the end
            n_init (int): The number of random sampling before using TPE
            max_evals (int): The number of total evaluations.
            seed (int): The random seed.
            metric_name (str): The name of the metric (or that of the objective function value)
            runtime_name (str): The name of the runtime metric.
            only_requirements (bool): If True, we only save runtime and loss.
            n_ei_candidates (int): The number of samplings to optimize the EI value
            result_keys (list[str]): Keys of results.
            min_bandwidth_factor (float): The minimum bandwidth for numerical parameters
            top (float): The hyperparam of the cateogircal kernel. It defines the prob of the top category.
        """
        result_keys = result_keys if result_keys is not None else [metric_name]
        super().__init__(
            obj_func=obj_func,
            config_space=config_space,
            resultfile=resultfile,
            n_init=n_init,
            max_evals=max_evals,
            seed=seed,
            metric_name=metric_name,
            runtime_name=runtime_name,
            only_requirements=only_requirements,
            result_keys=result_keys,
        )

        self._tpe_samplers = {
            key: TreeStructuredParzenEstimator(
                config_space=config_space,
                metric_name=key,
                runtime_name=runtime_name,
                n_ei_candidates=n_ei_candidates,
                seed=seed,
                min_bandwidth_factor=min_bandwidth_factor,
                top=top,
                percentile_func=percentile_func_maker(),
            )
            for key in result_keys
        }

    def update(self, eval_config: dict[str, Any], results: dict[str, float], runtime: float) -> None:
        for key, val in results.items():
            self._tpe_samplers[key].update_observations(eval_config=eval_config, loss=val, runtime=runtime)

    def fetch_observations(self) -> dict[str, np.ndarray]:
        observations = self._tpe_samplers[self._metric_name].observations
        for key in self._result_keys:
            observations[key] = self._tpe_samplers[key].observations[key]

        return observations

    def _get_config_cands(self, n_samples_dict: dict[str, int]) -> dict[str, np.ndarray]:
        config_cands: dict[str, np.ndarray] = {}
        for key in self._result_keys:
            tpe_sampler = self._tpe_samplers[key]
            n_samples = n_samples_dict.get(key, tpe_sampler._n_ei_candidates)
            if n_samples == 0:
                continue

            configs = tpe_sampler.get_config_candidates(n_samples)
            if len(config_cands) == 0:
                config_cands = configs
            else:
                config_cands = {
                    hp_name: np.concatenate([config_cands[hp_name], configs[hp_name]]) for hp_name in configs.keys()
                }

        return config_cands

    def _compute_probability_improvement(
        self, config_cands: dict[str, np.ndarray], weight_dict: dict[str, float]
    ) -> np.ndarray:
        pi_config_array = np.zeros((len(self._result_keys), config_cands[self._hp_names[0]].size))
        weights = np.ones(len(self._result_keys))

        for i, key in enumerate(self._result_keys):
            tpe_sampler = self._tpe_samplers[key]
            pi_config_array[i] += tpe_sampler.compute_probability_improvement(config_cands=config_cands)
            weights[i] = weight_dict[key]

        return weights @ pi_config_array

    def sample(
        self, weight_dict: dict[str, float] | None = None, n_samples_dict: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """
        Sample a configuration using tree-structured parzen estimator (TPE)

        Args:
            weights (dict[str, float] | None):
                Weights for each tpe samplers.
            n_samples_dict (dict[str, int] | None):
                The number of samples for each tpe samplers.

        Returns:
            eval_config (dict[str, Any]): A sampled configuration from TPE
        """
        n_samples_dict = {} if n_samples_dict is None else n_samples_dict
        weight_dict = {key: 1.0 for key in self._result_keys} if weight_dict is None else weight_dict

        config_cands = self._get_config_cands(n_samples_dict)
        pi_config = self._compute_probability_improvement(config_cands, weight_dict)
        best_idx = int(np.argmax(pi_config))
        eval_config = {hp_name: config_cands[hp_name][best_idx] for dim, hp_name in enumerate(self._hp_names)}

        return self._revert_eval_config(eval_config=eval_config)
