from typing import Dict
import os

import json

from util.utils import get_config_space, get_logger, ParameterSettings
from optimizer.tpe import TPEOptimizer

from hpolib.tabular_benchmark import (
    FCNetNavalPropulsionBenchmark,
    FCNetParkinsonsTelemonitoringBenchmark,
    FCNetProteinStructureBenchmark,
    FCNetSliceLocalizationBenchmark
)


if __name__ == '__main__':
    js = open('hpolib/params.json')
    searching_space: Dict[str, ParameterSettings] = json.load(js)
    config_space = get_config_space(searching_space, hp_module_path='hpolib')

    logger = get_logger(file_name='hpolib', logger_name='hpolib')
    benchmark = [
        FCNetNavalPropulsionBenchmark,
        FCNetParkinsonsTelemonitoringBenchmark,
        FCNetProteinStructureBenchmark,
        FCNetSliceLocalizationBenchmark
    ][0]

    # You need to change the path according to your path to the data
    data_dir = f'{os.environ["HOME"]}/research/nas_benchmarks/fcnet_tabular_benchmarks/'

    bm = benchmark(data_dir=data_dir)
    obj_func = bm.objective_func

    opt = TPEOptimizer(
        obj_func=obj_func,
        config_space=config_space,
        mutation_prob=0.05,
        resultfile='hpolib'
    )
    opt.optimize(logger)
    logger.info(f'The oracle of this benchmark: {bm.get_best_configuration()}')
