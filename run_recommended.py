from abc import ABCMeta
from typing import Optional
import os

import ConfigSpace as CS

from tpe.optimizer import TPEOptimizer
from tpe.utils.benchmarks import (
    Sphere,
    Styblinski,
    Rastrigin,
    Schwefel,
    Ackley,
    Griewank,
    Perm,
    KTablet,
    WeightedSphere,
    Rosenbrock,
    Levy,
    XinSheYang,
)
from tpe.utils.constants import QuantileFunc
from tpe.utils.tabular_benchmarks import AbstractBench, HPOBench, HPOLib, JAHSBench201

from experiment_utils import exist_file, save_observations


FUNCS = [
    Sphere,
    Styblinski,
    Rastrigin,
    Schwefel,
    Ackley,
    Griewank,
    Perm,
    KTablet,
    WeightedSphere,
    Rosenbrock,
    Levy,
    XinSheYang,
]
FUNCS += [HPOBench(dataset_id=i, seed=None) for i in range(8)]
FUNCS += [HPOLib(dataset_id=i, seed=None) for i in range(4)]
FUNCS += [JAHSBench201(dataset_id=i) for i in range(3)]
N_SEEDS = 10
MAX_EVALS = 200
N_INIT = 200 * 5 // 100
LINEAR, SQRT = "linear", "sqrt"


def collect_data(bench: AbstractBench, dim: Optional[int] = None) -> None:
    dir_name = "recommended"
    bench_name = bench().__class__.__name__ if isinstance(bench, ABCMeta) else bench.dataset_name
    bench_name = f"{bench_name}_{dim:0>2}d" if dim is not None else bench_name
    file_name = f"{bench_name}.json"
    if exist_file(dir_name, file_name):
        print(f"Skip {bench_name}")
        return

    config_space = CS.ConfigurationSpace()
    if dim is not None:
        config_space.add_hyperparameters(
            [CS.UniformFloatHyperparameter(f"x{d}", lower=-1, upper=1) for d in range(dim)]
        )

    results = []
    print(f"Collect {bench_name}")
    for seed in range(N_SEEDS):
        if hasattr(bench, "reseed"):
            bench.reseed(seed)

        opt = TPEOptimizer(
            obj_func=bench.func if isinstance(bench, ABCMeta) else bench,
            config_space=config_space if isinstance(bench, ABCMeta) else bench.config_space,
            max_evals=MAX_EVALS,
            n_init=N_INIT,
            weight_func_choice="older-smaller",
            quantile_func=QuantileFunc(choice="linear", alpha=0.1),
            multivariate=True,
            min_bandwidth_factor_for_discrete=None,
            top=0.8,
            seed=seed,
            resultfile=os.path.join(dir_name, file_name),
        )
        opt.optimize()
        results.append(opt.fetch_observations()["loss"].tolist())

    save_observations(dir_name=dir_name, file_name=file_name, data=results)


if __name__ == "__main__":
    for bench in FUNCS:
        if isinstance(bench, ABCMeta):
            for d in [5, 10, 30]:
                collect_data(bench, dim=d)
        else:
            collect_data(bench=bench)
