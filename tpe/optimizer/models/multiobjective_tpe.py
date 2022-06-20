from typing import Dict, List, Optional

import ConfigSpace as CS

from fast_pareto import is_pareto_front, nondominated_rank

import numpy as np

from tpe.optimizer.models import BaseTPE


class MultiObjectiveTPE(BaseTPE):
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        n_ei_candidates: int,
        metric_names: List[str],
        runtime_name: str,
        seed: Optional[int],
        min_bandwidth_factor: float,
        top: float,
    ):
        super().__init__(
            config_space=config_space,
            n_ei_candidates=n_ei_candidates,
            metric_names=metric_names,
            runtime_name=runtime_name,
            seed=seed,
            min_bandwidth_factor=min_bandwidth_factor,
            top=top,
        )
        self._n_fronts: int
        self._nondominated_ranks: np.ndarray

    def _percentile_func(self) -> int:
        n_observations = self._observations[self._metric_names[0]].size
        return max(self._n_fronts, int(np.ceil(0.15 * n_observations)))

    def _calculate_order(self, results: Optional[Dict[str, float]] = None) -> np.ndarray:
        with_new_result = results is not None

        n_observations = self._observations[self._metric_names[0]].size
        n_objectives = len(self._metric_names)
        costs = np.zeros((n_observations + with_new_result, n_objectives))
        for idx, metric_name in enumerate(self._metric_names):
            if not with_new_result:
                costs[:, idx] = self._observations[metric_name]
                continue
            else:
                costs[:-1, idx] = self._observations[metric_name]

            assert results is not None  # mypy redefinition
            new_loss = results.get(metric_name, None)
            if new_loss is None:
                raise ValueError(f"The evaluation must return {metric_name}.")

            costs[-1, idx] = new_loss

        self._nondominated_ranks = nondominated_rank(costs, tie_break=True)
        self._n_fronts = np.sum(is_pareto_front(costs))
        self._order = np.argsort(self._nondominated_ranks)
        return self._order
