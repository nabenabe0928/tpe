from typing import Any, Dict, List, Optional, Union

import ConfigSpace as CS

import numpy as np

from tpe.optimizer.base_optimizer import BaseOptimizer, ObjectiveFunc
from tpe.optimizer.models import ConstraintTPE, MetaLearnTPE, MultiObjectiveTPE, TPE


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
        runtime_name: str = "iter_time",
        only_requirements: bool = False,
        n_ei_candidates: int = 24,
        objective_names: List[str] = DEFAULT_OBJECTIVE_NAMES[:],
        constraints: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        min_bandwidth_factor: float = 1e-1,
        top: float = 1.0,
        warmstart_configs: Optional[Dict[str, np.ndarray]] = None,
        random_ratio: Optional[float] = None,
    ):
        """
        Args:
            obj_func (ObjectiveFunc): The objective function.
            config_space (CS.ConfigurationSpace): The searching space of the task
            resultfile (str): The name of the result file to output in the end
            n_init (int): The number of random sampling before using TPE
            max_evals (int): The number of total evaluations.
            seed (int): The random seed.
            objective_name (str): The name of the metric (or that of the objective function value)
            runtime_name (str): The name of the runtime metric.
            only_requirements (bool): If True, we only save runtime and loss.
            n_ei_candidates (int): The number of samplings to optimize the EI value
            objective_names (List[str]): The objective function names required for multi-objective.
            constraint_names (List[str]): The constraint names required for constraint optimization.
            metadata (Optional[Dict[str, Dict[str, np.ndarray]]]):
                Meta data required for meta-learning.
                Dict[task name, Dict[hyperparam/objective name, observation array]].
            min_bandwidth_factor (float): The minimum bandwidth for numerical parameters
            top (float): The hyperparam of the cateogircal kernel. It defines the prob of the top category.
            warmstart_configs (Optional[Dict[str, np.ndarray]]):
                The configurations that will be evaluated as the initialization.
            random_ratio (Optional[float]):
                How often we should sample a random configuration.
                In other words, epsillon in the epsillon-greedy algorithm.
                random_ratio of non-zero guarantees global optimization
                while random_ratio of zero (or None) is suitable for local optimization.
        """
        super().__init__(
            obj_func=obj_func,
            config_space=config_space,
            resultfile=resultfile,
            n_init=0 if warmstart_configs is not None else int(n_init),
            max_evals=max_evals,
            seed=seed,
            constraints=constraints,
            objective_names=objective_names,
            runtime_name=runtime_name,
            only_requirements=only_requirements,
        )

        tpe_params = dict(
            config_space=config_space,
            n_ei_candidates=n_ei_candidates,
            runtime_name=runtime_name,
            seed=seed,
            min_bandwidth_factor=min_bandwidth_factor,
            top=top,
        )
        self._sampler: Union[TPE, ConstraintTPE, MetaLearnTPE, MultiObjectiveTPE]
        self._warmstart_configs = warmstart_configs
        self._n_warmstart_configs = 0 if warmstart_configs is None else warmstart_configs[self._hp_names[0]].size
        self._warmstart_counter = 0
        self._random_ratio = random_ratio if random_ratio is not None else 0.0
        self._init_samplers(
            objective_names=objective_names, constraints=constraints, metadata=metadata, tpe_params=tpe_params
        )

    def _init_samplers(
        self,
        objective_names: List[str],
        metadata: Optional[Dict[str, Dict[str, np.ndarray]]],
        constraints: Optional[Dict[str, float]],
        tpe_params: Dict[str, Any],
    ) -> None:
        if metadata is not None:
            self._sampler = MetaLearnTPE(
                objective_names=objective_names, constraints=constraints, metadata=metadata, **tpe_params
            )
        elif constraints is not None:
            self._sampler = ConstraintTPE(objective_names=objective_names, constraints=constraints, **tpe_params)
        elif len(objective_names) == 1:
            self._sampler = TPE(objective_name=objective_names[0], **tpe_params)
        else:
            self._sampler = MultiObjectiveTPE(objective_names=objective_names, **tpe_params)

    def _validate_hyperparameters(self, observations: Dict[str, np.ndarray]) -> None:
        for hp_name in self._config_space:
            config = self._config_space.get_hyperparameter(hp_name)
            unique_vals = np.unique(observations[hp_name])
            if not hasattr(config, "choices") and not hasattr(config, "sequence"):
                EPS = 1e-12
                lb, ub = config.lower, config.upper
                if np.any((lb - EPS > unique_vals) | (unique_vals > ub + EPS)):
                    raise ValueError(f"Provided observations must be in the specified range ({lb}, {ub})")
            else:
                possible_vals = config.choices if hasattr(config, "choices") else config.sequence
                unknown_vals = set(unique_vals) - set(possible_vals)
                if len(unknown_vals) > 0:
                    raise ValueError(f"Provided observations include unknown values {unknown_vals}")

    def apply_knowledge_augmentation(self, observations: Dict[str, np.ndarray]) -> None:
        self._validate_hyperparameters(observations)
        self._sampler.apply_knowledge_augmentation(observations)

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
        if self._warmstart_configs is not None and self._warmstart_counter < self._n_warmstart_configs:
            return self._warmstart_sample()

        if self._rng.random() < self._random_ratio:  # random policy for the epsillon-greedy
            return self.initial_sample()

        config_cands = self._sampler.get_config_candidates()
        pi_config = self._compute_probability_improvement(config_cands=config_cands)
        best_idx = int(np.argmax(pi_config))
        eval_config = {hp_name: config_cands[hp_name][best_idx] for dim, hp_name in enumerate(self._hp_names)}
        return self._revert_eval_config(eval_config)

    def _warmstart_sample(self) -> Dict[str, Any]:
        idx = self._warmstart_counter
        assert self._warmstart_configs is not None  # mypy re-definition
        eval_config = {hp_name: self._warmstart_configs[hp_name][idx] for hp_name in self._hp_names}
        self._warmstart_counter += 1
        return eval_config
