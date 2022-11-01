from typing import Any, Callable, Dict, Literal, Optional

import ConfigSpace as CS

import numpy as np

from tpe.optimizer.base_optimizer import BaseOptimizer, ObjectiveFunc
from tpe.optimizer.tpe import TreeStructuredParzenEstimator
from tpe.utils.constants import QuantileFunc, WeightFuncs


class TPEOptimizer(BaseOptimizer):
    def __init__(
        self,
        obj_func: ObjectiveFunc,
        config_space: CS.ConfigurationSpace,
        resultfile: str = "temp",
        n_init: int = 10,
        max_evals: int = 100,
        seed: Optional[int] = None,
        metric_name: str = "loss",
        only_requirements: bool = True,
        n_ei_candidates: int = 24,
        min_bandwidth_factor: float = 1e-1,
        min_bandwidth_factor_for_discrete: float = 1.0,
        top: float = 1.0,
        quantile_func: Callable[[int], int] = QuantileFunc(),
        weight_func_choice: Literal["uniform", "older-smaller", "weaker-smaller", "expected-improvement"] = "uniform",
        multivariate: bool = True,
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
            min_bandwidth_factor (float): The minimum bandwidth for numerical parameters
            top (float): The hyperparam of the cateogircal kernel. It defines the prob of the top category.
            multivariate (bool): Whether to use multivariate kernel or not.
        """
        super().__init__(
            obj_func=obj_func,
            config_space=config_space,
            resultfile=resultfile,
            n_init=n_init,
            max_evals=max_evals,
            seed=seed,
            metric_name=metric_name,
            only_requirements=only_requirements,
        )

        self._tpe_sampler = TreeStructuredParzenEstimator(
            config_space=config_space,
            metric_name=metric_name,
            n_ei_candidates=n_ei_candidates,
            seed=seed,
            min_bandwidth_factor=min_bandwidth_factor,
            min_bandwidth_factor_for_discrete=min_bandwidth_factor_for_discrete,
            top=top,
            quantile_func=quantile_func,
            weight_func=WeightFuncs(choice=weight_func_choice),
            multivariate=multivariate,
        )

    def update(self, eval_config: Dict[str, Any], loss: float) -> None:
        self._tpe_sampler.update_observations(eval_config=eval_config, loss=loss)

    def fetch_observations(self) -> Dict[str, np.ndarray]:
        return self._tpe_sampler.observations

    def _get_config_cands(self) -> Dict[str, np.ndarray]:
        return self._tpe_sampler.get_config_candidates()

    def _compute_probability_improvement(self, config_cands: Dict[str, np.ndarray]) -> np.ndarray:
        return self._tpe_sampler.compute_probability_improvement(config_cands=config_cands)

    def sample(self) -> Dict[str, Any]:
        """
        Sample a configuration using tree-structured parzen estimator (TPE)

        Returns:
            eval_config (Dict[str, Any]): A sampled configuration from TPE
        """
        config_cands = self._get_config_cands()
        pi_config = self._compute_probability_improvement(config_cands)

        eval_config: Dict[str, Any] = {}
        if self._tpe_sampler._multivariate:
            best_idx = int(np.argmax(pi_config))
            eval_config = {hp_name: config_cands[hp_name][best_idx] for dim, hp_name in enumerate(self._hp_names)}
        else:
            best_indices = np.argmax(pi_config, axis=-1)
            eval_config = {
                hp_name: config_cands[hp_name][best_indices[dim]] for dim, hp_name in enumerate(self._hp_names)
            }

        return self._revert_eval_config(eval_config=eval_config)
