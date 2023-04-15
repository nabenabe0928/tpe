from typing import Dict

import ConfigSpace as CS

from dehb import DEHB

from tpe.utils.simulator_wrapper import CentralWorker


def obj_func(config: Dict, budget: float) -> Dict:
    res = {
        "fitness": config["x"] ** 2,
        "cost": budget / 10.0,
    }
    return res


if __name__ == "__main__":
    n_workers = 4
    max_budget = 100
    max_evals = 100
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5, 5))
    worker = CentralWorker(
        obj_func=obj_func,
        n_workers=n_workers,
        max_budget=max_budget,
        max_evals=max_evals,
        subdir_name="dehb",
        loss_key="fitness",
        runtime_key="cost",
    )

    dehb = DEHB(
        f=worker,
        cs=config_space,
        dimensions=1,
        min_budget=1,
        max_budget=max_budget,
        eta=3,
        client=None,
        n_workers=n_workers,
        output_path="dehb-log/"
    )
    dehb.run(fevals=max_evals)
