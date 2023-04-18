from typing import Any, Dict

import neps

from baseline_utils._neps import run_neps


def obj_func(eval_config: Dict[str, Any], budget: int) -> Dict[str, float]:
    output = dict(loss=eval_config["x"]**2, cost=budget / 100)
    print(output)
    return output


pipeline_space = {
    "x": neps.FloatParameter(lower=-5.0, upper=5.0),
    "epoch": neps.IntegerParameter(lower=1, upper=100, is_fidelity=True),
}
run_neps(
    obj_func=obj_func,
    pipeline_space=pipeline_space,
    min_budget=1,
    max_budget=100,
    n_workers=4,
    subdir_name="neps",
    budget_key="epoch",
)
