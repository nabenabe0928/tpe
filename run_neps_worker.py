from typing import Any, Dict

import neps

from tpe.utils.simulator_wrapper import ObjectiveFunc, WorkerFunc


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


BUDGET_KEY = "epoch"
max_evals = 100
max_budget = 100


def obj_func(eval_config: Dict[str, Any], budget: int) -> Dict[str, float]:
    output = dict(loss=eval_config["x"]**2, cost=budget / 100)
    print(output)
    return output


pipeline_space = {
    "x": neps.FloatParameter(lower=-5.0, upper=5.0),
    BUDGET_KEY: neps.IntegerParameter(lower=1, upper=max_budget, is_fidelity=True),
}

worker = NEPSWorker(
    subdir_name="neps",
    n_workers=4,
    func=obj_func,
    max_budget=max_budget,
    budget_key=BUDGET_KEY,
)

neps.run(
    run_pipeline=worker,
    pipeline_space=pipeline_space,
    root_directory="neps-log",
    max_cost_total=1.0,
    # max_evaluations_total=max_evals,
)

worker.finish()
