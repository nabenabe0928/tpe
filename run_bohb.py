from typing import Any, Dict

import ConfigSpace as CS

from baseline_utils._bohb import run_bohb


def obj_func(eval_config: Dict[str, Any], budget: int) -> Dict[str, float]:
    return dict(loss=eval_config["x"]**2, runtime=budget / 10.0)


config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5.0, 5.0))

run_bohb(
    obj_func=obj_func,
    config_space=config_space,
    min_budget=1,
    max_budget=100,
    run_id="test-bohb",
    n_workers=4,
    subdir_name="bohb",
)
