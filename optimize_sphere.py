import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from util.utils import get_logger
import numpy as np

from optimizer.tpe import TPEOptimizer


def sphere(eval_config):
    vals = np.array(list(eval_config.values()))
    return (vals ** 2).sum()


if __name__ == '__main__':
    dim = 10
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f'x{d}', lower=-5, upper=5))

    logger = get_logger(file_name='sphere', logger_name='sphere')
    opt = TPEOptimizer(obj_func=sphere, config_space=cs, mutation_prob=0.0, resultfile='sphere')
    opt.optimize(logger)
