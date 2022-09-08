from typing import Any, Dict, List, Optional, Type, Union

import ConfigSpace as CS

import numpy as np

from task_similarity import IoUTaskSimilarity

from tpe.optimizer.models import ConstraintTPE, MultiObjectiveTPE, TPE
from tpe.optimizer.models.base_tpe import AbstractTPE
from tpe.utils.constants import NumericType, OBJECTIVE_KEY
from tpe.utils.special_funcs import exp


# TPESamplerType = Union[ConstraintTPE, TPE, MultiObjectiveTPE]
TPESamplerType = Union[TPE, MultiObjectiveTPE]


class MetaLearnTPE(AbstractTPE):
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        n_ei_candidates: int,
        objective_names: List[str],
        runtime_name: str,
        seed: Optional[int],
        min_bandwidth_factor: float,
        top: float,
        metadata: Dict[str, Dict[str, np.ndarray]],
        constraints: Optional[Dict[str, float]] = None,
        n_samples: int = 200,
        dim_reduction_factor: float = 5.0,
    ):
        if OBJECTIVE_KEY in metadata:
            raise KeyError(f"metadata cannot include the key name {OBJECTIVE_KEY}")
        if constraints is not None:
            raise NotImplementedError("Meta-learning for constraint optimization is not supported")

        tpe_params = dict(
            config_space=config_space,
            n_ei_candidates=n_ei_candidates,
            runtime_name=runtime_name,
            seed=seed,
            min_bandwidth_factor=min_bandwidth_factor,
            top=top,
        )
        self._dim_reduction_factor = dim_reduction_factor
        self._source_task_hp_importance: Optional[Dict[str, np.ndarray]] = None
        self._objective_names = objective_names[:]
        self._n_samples = n_samples
        self._rng = np.random.RandomState(seed)
        self._config_space = config_space
        self._samplers: Dict[str, TPESamplerType] = {}
        self._task_names = [OBJECTIVE_KEY] + list(metadata.keys())
        self._init_samplers(
            objective_names=objective_names, constraints=constraints, metadata=metadata, tpe_params=tpe_params
        )

    def _init_samplers(
        self,
        objective_names: List[str],
        metadata: Dict[str, Dict[str, np.ndarray]],
        constraints: Optional[Dict[str, float]],
        tpe_params: Dict[str, Any],
    ) -> None:
        sampler_class: Type[Union[TPE, ConstraintTPE, MultiObjectiveTPE]]
        if constraints is not None:
            # tpe_params.update(objective_names=objective_names, constraints=constraints)
            # sampler_class = ConstraintTPE
            raise NotImplementedError
        elif len(objective_names) == 1:
            tpe_params.update(objective_name=objective_names[0])
            sampler_class = TPE
        else:
            tpe_params.update(objective_names=objective_names)
            sampler_class = MultiObjectiveTPE

        self._samplers[OBJECTIVE_KEY] = sampler_class(**tpe_params)
        for task_name, data in metadata.items():
            sampler = sampler_class(**tpe_params)
            sampler.apply_knowledge_augmentation(data)
            self._samplers[task_name] = sampler

    def update_observations(
        self, eval_config: Dict[str, NumericType], results: Dict[str, float], runtime: float
    ) -> None:
        self._samplers[OBJECTIVE_KEY].update_observations(eval_config=eval_config, results=results, runtime=runtime)

    def apply_knowledge_augmentation(self, observations: Dict[str, np.ndarray]) -> None:
        self._samplers[OBJECTIVE_KEY].apply_knowledge_augmentation(observations)

    def _compute_task_similarity(self) -> np.ndarray:
        observations_set: List[Dict[str, np.ndarray]] = [
            self._samplers[task_name].observations for task_name in self._task_names
        ]
        ts = IoUTaskSimilarity(
            n_samples=self._n_samples,
            config_space=self._config_space,
            observations_set=observations_set,
            objective_names=self._objective_names,
            promising_quantile=0.10,  # same as in mo-tpe, and this must be adapted
            rng=self._rng,
            dim_reduction_factor=self._dim_reduction_factor,
            source_task_hp_importance=self._source_task_hp_importance,
        )
        if self._source_task_hp_importance is None:
            self._source_task_hp_importance = ts.source_task_hp_importance

        n_tasks = len(self._task_names)
        sim = ts.compute(task_pairs=[(0, i) for i in range(1, n_tasks)])
        assert isinstance(sim[0], np.ndarray)  # mypy re-definition
        return sim[0]

    def _compute_task_weights(self, sim: np.ndarray) -> np.ndarray:
        n_tasks = len(self._task_names)
        weights = np.zeros_like(sim)

        weights[0] = 1 - np.sum(sim[1:]) / n_tasks
        for i in range(1, n_tasks):
            weights[i] = sim[i] / n_tasks

        return weights

    def get_task_weights(self) -> np.ndarray:
        sim = self._compute_task_similarity()
        task_weights = self._compute_task_weights(sim)
        return task_weights

    def compute_probability_improvement(self, config_cands: Dict[str, np.ndarray]) -> np.ndarray:
        task_weights = self.get_task_weights()
        n_samples_in_lower, n_samples_in_upper = np.zeros_like(task_weights), np.zeros_like(task_weights)
        n_cands = config_cands[list(config_cands.keys())[0]].size
        taskwise_ll_lower, taskwise_ll_upper = np.zeros((2, task_weights.size, n_cands))

        for idx, sampler in enumerate(self._samplers.values()):
            ll_lower, ll_upper = sampler.compute_config_loglikelihoods(config_cands)
            n_samples_in_lower[idx] += sampler._mvpe_lower.size
            n_samples_in_upper[idx] += sampler._mvpe_upper.size
            taskwise_ll_lower[idx] += ll_lower
            taskwise_ll_upper[idx] += ll_upper

        task_weights_lower = task_weights * n_samples_in_lower
        task_weights_upper = task_weights * n_samples_in_upper
        return (task_weights_lower @ exp(taskwise_ll_lower)) / (task_weights_upper @ exp(taskwise_ll_upper))

    def get_config_candidates(self) -> Dict[str, np.ndarray]:
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

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return self._samplers[OBJECTIVE_KEY].observations
