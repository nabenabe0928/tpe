import time
from typing import Dict

import ConfigSpace as CS

from dehb import DEHB


def obj_func(config: Dict, budget: float) -> Dict:
    start = time.time()
    res = {
        "fitness": config["x"] ** 2,
        "cost": time.time() - start,
        "info": {"budget": budget}
    }
    return res


if __name__ == "__main__":
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5, 5))
    dehb = DEHB(
        f=obj_func,
        cs=config_space,
        dimensions=1,
        min_budget=1,
        max_budget=100,
        eta=3,
        client=None,
        n_workers=4,
        output_path="dehb-log/"
    )
    dehb.run(total_cost=1)
