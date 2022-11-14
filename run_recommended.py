from abc import ABCMeta
from typing import List, Optional
import os

import ConfigSpace as CS

import ujson as json

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

# Control params
WEIGHT_FUNC_CHOICE = ["uniform", "older-smaller", "weaker-smaller", "expected-improvement"][0]
QUANTILE_CHOICE = ["linear", "sqrt"][1]
ALPHA = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0][5]
MULTIVARIATE = [True, False][0]
BW_CONT = [0.01, 0.03, 0.1, 0.3][0]
BW_DISC = [0.25, 0.5, 1.0, 2.0, None][1]
TOP = [0.8, 0.9, 1.0, 2.0][0]

DIR_BENCH = "_".join(
    [
        f"multivariate={MULTIVARIATE}",
        f"quantile={QUANTILE_CHOICE}",
        f"alpha={ALPHA}",
        f"weight={WEIGHT_FUNC_CHOICE}",
        f"min_bandwidth_factor={BW_CONT}",
    ]
)
DIR_JAHS = "_".join(
    [
        f"multivariate={MULTIVARIATE}",
        f"quantile={QUANTILE_CHOICE}",
        f"alpha={ALPHA}",
        f"weight={WEIGHT_FUNC_CHOICE}",
        f"min_bandwidth_factor={BW_CONT}",
        f"min_bandwidth_factor_for_discrete={BW_DISC}",
        f"top={TOP}",
    ]
)
DIR_HPOBENCH = "_".join(
    [
        f"multivariate={MULTIVARIATE}",
        f"quantile={QUANTILE_CHOICE}",
        f"alpha={ALPHA}",
        f"weight={WEIGHT_FUNC_CHOICE}",
        f"min_bandwidth_factor_for_discrete={BW_DISC}",
    ]
)
DIR_HPOLIB = "_".join(
    [
        f"multivariate={MULTIVARIATE}",
        f"quantile={QUANTILE_CHOICE}",
        f"alpha={ALPHA}",
        f"weight={WEIGHT_FUNC_CHOICE}",
        f"min_bandwidth_factor_for_discrete={BW_DISC}",
        f"top={TOP}",
    ]
)
HPOLIB_NAMES = ["naval_propulsion", "parkinsons_telemonitoring", "slice_localization", "protein_structure"]
JAHS_NAMES = ["cifar10", "fashion_mnist", "colorectal_histology"]


def check_file(bench_name: str) -> Optional[List[List[float]]]:
    fn = f"{bench_name}.json"
    if any(bench_name.endswith(f"_{d}d") for d in ["05", "10", "30"]):
        dir_name = DIR_BENCH
    elif BW_DISC is not None:
        if bench_name in HPOLIB_NAMES:
            dir_name = DIR_HPOLIB
        elif bench_name in JAHS_NAMES:
            dir_name = DIR_JAHS
        else:
            dir_name = DIR_HPOBENCH
    else:
        return None

    path = os.path.join("results", dir_name, fn)
    return json.load(open(path))


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
    query = check_file(bench_name)
    if query is not None:
        print(f"Re-use the results for {bench_name}")
        save_observations(dir_name=dir_name, file_name=file_name, data=query)
        return

    print(f"Collect {bench_name}")
    for seed in range(N_SEEDS):
        if hasattr(bench, "reseed"):
            bench.reseed(seed)

        opt = TPEOptimizer(
            obj_func=bench.func if isinstance(bench, ABCMeta) else bench,
            config_space=config_space if isinstance(bench, ABCMeta) else bench.config_space,
            max_evals=MAX_EVALS,
            n_init=N_INIT,
            weight_func_choice=WEIGHT_FUNC_CHOICE,
            quantile_func=QuantileFunc(choice=QUANTILE_CHOICE, alpha=ALPHA),
            multivariate=MULTIVARIATE,
            min_bandwidth_factor=BW_CONT,
            min_bandwidth_factor_for_discrete=BW_DISC,
            top=TOP if TOP < 1.1 else None,
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
