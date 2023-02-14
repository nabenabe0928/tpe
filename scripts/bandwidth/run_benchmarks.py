import itertools
import json
import os
from argparse import ArgumentParser
from typing import Dict, List

import ConfigSpace as CS

from tpe.optimizer import TPEOptimizer
from tpe.utils.benchmarks import (
    AbstractFunc,
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


parser = ArgumentParser()
parser.add_argument("--dim", type=int, choices=[5, 10, 30], required=True)
args = parser.parse_args()
DIM = args.dim

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
N_SEEDS = 10
MAX_EVALS = 200
N_INIT = 200 * 5 // 100
LINEAR, SQRT = "linear", "sqrt"
CONFIG_SPACE = CS.ConfigurationSpace()
CONFIG_SPACE.add_hyperparameters([CS.UniformFloatHyperparameter(name=f"x{d}", lower=-1, upper=1) for d in range(DIM)])


def save_observations(dir_name: str, file_name: str, data: Dict[str, List[List[float]]]) -> None:
    path = os.path.join("results-bandwidth", dir_name)
    with open(os.path.join(path, file_name), mode="w") as f:
        json.dump(data, f, indent=4)


def exist_file(dir_name: str, file_name: str) -> bool:
    path = os.path.join("results-bandwidth", dir_name)
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        print(f"{file_path} exists; skip")
        return True
    else:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, file_name), mode="w"):
            pass

        return False


def collect_data(
    func_cls: AbstractFunc,
    multivariate: bool,
    choice: str,
    alpha: float,
    weight_func_choice: str,
    magic_clip: bool,
    heuristic: bool,
    min_bandwidth_factor: float,
) -> None:

    func_name = func_cls().__class__.__name__
    heuristic_name = heuristic if heuristic is not None else "scott"
    print(func_name, multivariate, choice, alpha, weight_func_choice, heuristic_name, min_bandwidth_factor)
    dir_name = "_".join(
        [
            f"multivariate={multivariate}",
            f"quantile={choice}",
            f"alpha={alpha}",
            f"weight={weight_func_choice}",
            f"magic-clip={magic_clip}",
            f"heuristic={heuristic_name}",
            f"min_bandwidth_factor={min_bandwidth_factor}",
        ]
    )
    file_name = f"{func_name}_{DIM:0>2}d.json"
    if exist_file(dir_name, file_name):
        return

    results = []
    for seed in range(N_SEEDS):
        opt = TPEOptimizer(
            obj_func=func_cls.func,
            config_space=CONFIG_SPACE,
            max_evals=MAX_EVALS,
            n_init=N_INIT,
            weight_func_choice=weight_func_choice,
            quantile_func=QuantileFunc(choice=choice, alpha=alpha),
            multivariate=multivariate,
            seed=seed,
            resultfile=os.path.join(dir_name, file_name),
            magic_clip=magic_clip,
            heuristic=heuristic,
            min_bandwidth_factor=min_bandwidth_factor,
        )
        opt.optimize()
        results.append(opt.fetch_observations()["loss"].tolist())

    save_observations(dir_name=dir_name, file_name=file_name, data=results)


if __name__ == "__main__":
    for params in itertools.product(
        *(
            [True, False],  # multivariate
            [
                (LINEAR, 0.15),
                (SQRT, 0.25),
                (SQRT, 1.0),
            ],  # quantile_func
            ["uniform", "older-smaller"],  # weight_func_choice
            [True, False],  # magic_clip
            [None, "optuna", "hyperopt"],  # heuristics
            [0.01, 0.03, 0.1, 0.3],  # min bandwidth factor
            FUNCS,
        )
    ):
        collect_data(
            func_cls=params[-1],
            multivariate=params[0],
            choice=params[1][0],
            alpha=params[1][1],
            weight_func_choice=params[2],
            magic_clip=params[3],
            heuristic=params[4],
            min_bandwidth_factor=params[5],
        )
