from typing import Any, Callable, Dict, Optional

import numpy as np

import ConfigSpace as CS

from optimizer.base_optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    def __init__(self, obj_func: Callable, config_space: CS.ConfigurationSpace,
                 resultfile: str, n_init: int = 10, max_evals: int = 100,
                 seed: Optional[int] = None, metric_name: str = 'loss'):

        super().__init__(
            obj_func=obj_func,
            config_space=config_space,
            resultfile=resultfile,
            n_init=n_init,
            max_evals=max_evals,
            seed=seed,
            metric_name=metric_name
        )

        self._observations = {metric_name: np.array([])}
        self._observations.update({hp_name: np.array([]) for hp_name in self._hp_names})

    def update(self, eval_config: Dict[str, Any], loss: float) -> None:
        for hp_name, val in eval_config.items():
            self._observations[hp_name] = np.append(self._observations[hp_name], val)

        self._observations[self._metric_name] = np.append(self._observations[self._metric_name], loss)

    def fetch_observations(self) -> Dict[str, np.ndarray]:
        return {hp_name: vals.copy() for hp_name, vals in self._observations.items()}

    def sample(self) -> Dict[str, Any]:
        return self.initial_sample()
