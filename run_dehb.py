from typing import Dict

import ConfigSpace as CS

from baseline_utils._dehb import run_dehb


def obj_func(config: Dict, budget: float) -> Dict:
    res = {
        "fitness": config["x"] ** 2,
        "cost": budget / 10.0,
    }
    return res


if __name__ == "__main__":
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5, 5))
    run_dehb(
        obj_func=obj_func,
        config_space=config_space,
        min_budget=1,
        max_budget=100,
        n_workers=4,
        subdir_name="dehb",
    )
