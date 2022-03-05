from argparse import ArgumentParser

from util.utils import get_logger
from optimizer.tpe import TPEOptimizer

from targets.nasbench101.api import DatasetChoices, NASBench101, SearchSpaceChoices


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--search_space', choices=[f'cifar10{c}' for c in ['A', 'B', 'C']],
                        default='cifar10A')

    args = parser.parse_args()
    search_space = args.search_space
    search_space_choice = getattr(SearchSpaceChoices, search_space)
    name = f'nasbench101_{search_space}'
    logger = get_logger(file_name=name, logger_name=name)

    bm = NASBench101(dataset=DatasetChoices.only108, search_space=search_space_choice)

    obj_func = bm.objective_func

    opt = TPEOptimizer(
        obj_func=obj_func,
        config_space=bm.config_space,
        resultfile=name
    )
    opt.optimize(logger)
