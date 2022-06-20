from typing import Any, Dict, List, Optional, Union

import ConfigSpace as CS

import numpy as np

from tpe.optimizer.base_optimizer import BaseOptimizer, ObjectiveFunc
from tpe.optimizer.models import MultiObjectiveTPE, TPE


DEFAULT_OBJECTIVE_NAMES = ["loss"]


class TPEOptimizer(BaseOptimizer):
    def __init__(
        self,
        obj_func: ObjectiveFunc,
        config_space: CS.ConfigurationSpace,
        *,
        resultfile: str = "temp",
        n_init: int = 10,
        max_evals: int = 100,
        seed: Optional[int] = None,
        metric_names: List[str] = DEFAULT_OBJECTIVE_NAMES[:],
        runtime_name: str = "iter_time",
        only_requirements: bool = False,
        n_ei_candidates: int = 24,
        result_keys: List[str] = DEFAULT_OBJECTIVE_NAMES[:],
        objective_names: List[str] = DEFAULT_OBJECTIVE_NAMES[:],
        constraint_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        min_bandwidth_factor: float = 1e-1,
        top: float = 1.0,
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
            result_keys (List[str]): Keys of results.
            objective_names (List[str]): The objective function names required for multi-objective.
            constraint_names (List[str]): The constraint names required for constraint optimization.
            metadata (Optional[Dict[str, Dict[str, np.ndarray]]]):
                Meta data required for meta-learning.
                Dict[task name, Dict[hyperparam/objective name, observation array]].
            min_bandwidth_factor (float): The minimum bandwidth for numerical parameters
            top (float): The hyperparam of the cateogircal kernel. It defines the prob of the top category.
        """
        super().__init__(
            obj_func=obj_func,
            config_space=config_space,
            resultfile=resultfile,
            n_init=n_init,
            max_evals=max_evals,
            seed=seed,
            # TODO: Adapt here
            metric_name=metric_names[0],
            runtime_name=runtime_name,
            only_requirements=only_requirements,
            result_keys=result_keys,
        )

        tpe_params = dict(
            config_space=config_space,
            n_ei_candidates=n_ei_candidates,
            runtime_name=runtime_name,
            seed=seed,
            min_bandwidth_factor=min_bandwidth_factor,
            top=top,
        )
        self._sampler: Union[TPE, MultiObjectiveTPE]
        if len(metric_names) == 1:
            self._sampler = TPE(metric_name=metric_names[0], **tpe_params)
        else:
            self._sampler = MultiObjectiveTPE(metric_names=metric_names, **tpe_params)

    def update(self, eval_config: Dict[str, Any], results: Dict[str, float], runtime: float) -> None:
        self._sampler.update_observations(eval_config=eval_config, results=results, runtime=runtime)

    def fetch_observations(self) -> Dict[str, np.ndarray]:
        observations = self._sampler.observations
        return observations

    def _compute_probability_improvement(self, config_cands: Dict[str, np.ndarray]) -> np.ndarray:
        return self._sampler.compute_probability_improvement(config_cands=config_cands)

    def sample(self) -> Dict[str, Any]:
        """
        Sample a configuration using tree-structured parzen estimator (TPE)

        Returns:
            eval_config (Dict[str, Any]): A sampled configuration from TPE
        """
        config_cands = self._sampler.get_config_candidates()
        pi_config = self._compute_probability_improvement(config_cands=config_cands)
        best_idx = int(np.argmax(pi_config))
        eval_config = {hp_name: config_cands[hp_name][best_idx] for dim, hp_name in enumerate(self._hp_names)}
        return self._revert_eval_config(eval_config)
