from typing import Any, Dict

import neps

from tpe.utils.simulator_wrapper import ObjectiveFunc, WorkerFunc
from tpe.utils.utils import AbstractObjFunc


class NEPSWorker(WorkerFunc):
    def __init__(
        self,
        subdir_name: str,
        n_workers: int,
        func: ObjectiveFunc,
        max_budget: int,
        budget_key: str,
        loss_key: str = "loss",
        runtime_key: str = "cost",
    ):
        super().__init__(
            subdir_name=subdir_name,
            n_workers=n_workers,
            func=func,
            max_budget=max_budget,
            loss_key=loss_key,
            runtime_key=runtime_key,
        )
        self._budget_key = budget_key

    def __call__(self, **eval_config: Dict[str, Any]) -> Dict[str, float]:
        _eval_config = eval_config.copy()
        budget = _eval_config.pop(self._budget_key)
        return super().__call__(_eval_config, budget)


def run_neps(
    obj_func: AbstractObjFunc,
    pipeline_space: Dict[str, neps.search_spaces.parameter.Parameter],
    min_budget: int,
    max_budget: int,
    n_workers: int,
    subdir_name: str,
    budget_key: str,
    max_evals: int = 450,  # eta=3,S=2,100 full evals
):
    worker = NEPSWorker(
        subdir_name="neps",
        n_workers=n_workers,
        func=obj_func,
        max_budget=max_budget,
        budget_key=budget_key,
    )

    neps.run(
        run_pipeline=worker,
        pipeline_space=pipeline_space,
        root_directory="neps-log",
        max_evaluations_total=max_evals + 10,
    )

    worker.finish()
