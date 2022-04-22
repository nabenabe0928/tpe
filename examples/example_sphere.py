import time
from typing import Dict, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.optimizer import TPEOptimizer
from tpe.utils.utils import get_logger


def sphere(eval_config: Dict[str, float]) -> Tuple[float, float]:
    start = time.time()
    vals = np.array(list(eval_config.values()))
    vals *= vals
    return np.sum(vals), time.time() - start


if __name__ == "__main__":
    dim = 10
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-5, upper=5))

    logger = get_logger(file_name="sphere", logger_name="sphere", disable=True)
    opt = TPEOptimizer(obj_func=sphere, config_space=cs, resultfile="sphere")
    opt.optimize(logger)
