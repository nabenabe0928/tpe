from typing import Any, Dict, List, Optional, Type, Union

import ConfigSpace as CS

import numpy as np

from tpe.optimizer.models import ConstraintTPE, MultiObjectiveTPE, TPE
from tpe.optimizer.models.base_tpe import AbstractTPE
from tpe.utils.constants import NumericType, OBJECTIVE_KEY


TPESamplerType = Union[ConstraintTPE, TPE, MultiObjectiveTPE]


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
    ):
        if OBJECTIVE_KEY in metadata:
            raise KeyError(f"metadata cannot include the key name {OBJECTIVE_KEY}")

        tpe_params = dict(
            config_space=config_space,
            n_ei_candidates=n_ei_candidates,
            runtime_name=runtime_name,
            seed=seed,
            min_bandwidth_factor=min_bandwidth_factor,
            top=top,
        )
        self._samplers: Dict[str, TPESamplerType] = {}
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
            tpe_params.update(objective_names=objective_names, constraints=constraints)
            sampler_class = ConstraintTPE
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

    def compute_probability_improvement(self, config_cands: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def get_config_candidates(self) -> Dict[str, np.ndarray]:
        pass

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return self._samplers[OBJECTIVE_KEY].observations
