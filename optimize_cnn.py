from typing import Dict

import json

from util.utils import get_config_space, get_logger, ParameterSettings
from cnn.train import get_objective_func
from optimizer.tpe import TPEOptimizer


if __name__ == '__main__':
    js = open('cnn/params.json')
    searching_space: Dict[str, ParameterSettings] = json.load(js)
    config_space = get_config_space(searching_space, hp_module_path='cnn')

    logger = get_logger(file_name='hoge', logger_name='cnn')
    obj_func = get_objective_func(
        logger=logger,
        searching_space=searching_space,
        config_space=config_space
    )

    opt = TPEOptimizer(
        obj_func=obj_func,
        config_space=config_space,
        mutation_prob=0.05,
        resultfile='cnn'
    )
    opt.optimize(logger)
