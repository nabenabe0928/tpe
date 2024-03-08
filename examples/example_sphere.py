import time
from typing import Dict, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.optimizer import TPEOptimizer


def sphere(eval_config: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    start = time.time()
    vals = np.array(list(eval_config.values()))
    vals *= vals
    return {"loss": np.sum(vals)}, time.time() - start


if __name__ == "__main__":
    dim = 10
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-5, upper=5))

    opt = TPEOptimizer(obj_func=sphere, config_space=cs, min_bandwidth_factor=1e-2, resultfile="sphere")
    print(opt.optimize(logger_name="sphere"))
