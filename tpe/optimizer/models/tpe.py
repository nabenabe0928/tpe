from typing import Dict, Optional

import ConfigSpace as CS

import numpy as np

from tpe.optimizer.models import BaseTPE


class TPE(BaseTPE):
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        n_ei_candidates: int,
        objective_name: str,
        runtime_name: str,
        seed: Optional[int],
        min_bandwidth_factor: float,
        top: float,
    ):
        super().__init__(
            config_space=config_space,
            n_ei_candidates=n_ei_candidates,
            objective_names=[objective_name],
            runtime_name=runtime_name,
            seed=seed,
            min_bandwidth_factor=min_bandwidth_factor,
            top=top,
        )
        self._objective_name = objective_name

    def _percentile_func(self) -> int:
        n_observations = self._observations[self._objective_name].size
        return int(np.ceil(0.25 * np.sqrt(n_observations)))

    def _calculate_order(self, results: Optional[Dict[str, float]] = None) -> np.ndarray:
        if results is None:
            loss_vals = self._observations[self._objective_name]
        else:
            loss_vals = np.append(self._observations[self._objective_name], results[self._objective_name])

        self._order = np.argsort(loss_vals)
        return self._order
