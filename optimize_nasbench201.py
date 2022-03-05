from util.utils import get_logger
from optimizer.tpe import TPEOptimizer

from targets.nasbench201.api import DatasetChoices, NASBench201


if __name__ == '__main__':
    logger = get_logger(file_name='nasbench201', logger_name='nasbench201')

    bm = NASBench201(dataset=DatasetChoices.imagenet)

    obj_func = bm.objective_func

    opt = TPEOptimizer(
        obj_func=obj_func,
        config_space=bm.config_space,
        resultfile='nasbench201'
    )
    opt.optimize(logger)
