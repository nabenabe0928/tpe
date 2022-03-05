from util.utils import get_logger
from optimizer.tpe import TPEOptimizer

from targets.hpolib.api import DatasetChoices, HPOBench


if __name__ == '__main__':
    logger = get_logger(file_name='hpolib', logger_name='hpolib')

    bm = HPOBench(dataset=DatasetChoices.protein_structure)

    obj_func = bm.objective_func

    opt = TPEOptimizer(
        obj_func=obj_func,
        config_space=bm.config_space,
        resultfile='hpolib'
    )
    opt.optimize(logger)
