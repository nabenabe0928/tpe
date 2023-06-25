import json
import os
from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional

import ConfigSpace as CS

import hyperopt

import numpy as np

import pickle

from tpe.optimizer import TPEOptimizer
from tpe.utils.benchmarks import (
    Ackley,
    DifferentPower,
    DixonPrice,
    Griewank,
    KTablet,
    Langermann,
    Levy,
    Michalewicz,
    Perm,
    Powell,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    Styblinski,
    Trid,
    WeightedSphere,
    XinSheYang,
)
from tpe.utils.tabular_benchmarks import HPOBench, HPOLib, JAHSBench201, LCBench, NASHPOBench2, OlympusBench


FUNCS = [
    Ackley,
    DifferentPower,
    DixonPrice,
    Griewank,
    KTablet,
    Langermann,
    Levy,
    Michalewicz,
    Perm,
    Powell,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    Styblinski,
    Trid,
    WeightedSphere,
    XinSheYang,
]
FUNCS += [HPOBench(dataset_id=i, seed=None) for i in range(8)]
FUNCS += [HPOLib(dataset_id=i, seed=None) for i in range(4)]
FUNCS += [JAHSBench201(dataset_id=i) for i in range(3)]
FUNCS += [LCBench(dataset_id=i) for i in range(8)]
FUNCS += [NASHPOBench2(dataset_id=0)]
FUNCS += [OlympusBench(dataset_id=i) for i in range(10)]


def wrapper_func(bench: Callable, config_space: CS.ConfigurationSpace) -> Callable:
    def func(eval_config: Dict[str, Any]) -> float:
        target = bench.func if isinstance(bench, ABCMeta) else bench
        return target(eval_config)

    return func


def get_init_configs(
    bench: Callable,
    config_space: CS.ConfigurationSpace,
    seed: int,
) -> List[Dict[str, Any]]:
    random_sampler = TPEOptimizer(
        obj_func=bench.func if isinstance(bench, ABCMeta) else bench,
        config_space=config_space,
        n_init=10,
        max_evals=10,
        seed=seed,
    )
    random_sampler.optimize()
    init_data = random_sampler.fetch_observations()
    return [{
        hp_name: int(init_data[hp_name][i]) if isinstance(init_data[hp_name][i], str) else init_data[hp_name][i]
        for hp_name in config_space} for i in range(10)
    ]


def extract_space(config_space: CS.ConfigurationSpace):
    space = {}
    for hp in config_space.get_hyperparameters():
        if isinstance(hp, CS.CategoricalHyperparameter):
            space[hp.name] = hyperopt.hp.choice(hp.name, [int(c) for c in hp.choices])
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            log = hp.log
            dist = hyperopt.hp.loguniform if log else hyperopt.hp.uniform
            lb, ub = np.log(hp.lower) if log else hp.lower, np.log(hp.upper) if log else hp.upper
            space[hp.name] = dist(hp.name, lb, ub)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            space[hp.name] = hyperopt.hp.randint(hp.name, hp.lower, hp.upper + 1)
        else:
            raise TypeError(f"{type(type(hp))} is not supported")

    return space


def collect_data(bench: Callable, dim: Optional[int] = None) -> None:
    bench_name = bench().__class__.__name__ if isinstance(bench, ABCMeta) else bench.dataset_name
    bench_name = f"{bench_name}_{dim:0>2}d" if dim is not None else bench_name
    print(bench_name, dim)

    config_space = CS.ConfigurationSpace()
    if dim is not None:
        config_space.add_hyperparameters(
            [CS.UniformFloatHyperparameter(f"x{d}", lower=-1, upper=1) for d in range(dim)]
        )

    dir_name = "results/hyperopt"
    os.makedirs(dir_name, exist_ok=True)
    path = os.path.join(dir_name, f"{bench_name}.json")
    if os.path.exists(path):
        print(f"{path} exists; skip")
        return
    else:
        with open(path, mode="w") as f:
            pass

    results = []
    for seed in range(10):
        if hasattr(bench, "reseed"):
            bench.reseed(seed)

        config_space = config_space if isinstance(bench, ABCMeta) else bench.config_space
        init_configs = get_init_configs(bench=bench, config_space=config_space, seed=seed)
        os.environ["HYPEROPT_FMIN_SEED"] = str(seed)
        save_loc = "hyperopt-output.pkl"
        hyperopt.fmin(
            fn=wrapper_func(bench=bench, config_space=config_space),
            space=extract_space(config_space),
            max_evals=190,
            points_to_evaluate=init_configs,
            trials_save_file=save_loc,
        )
        vals = [trial["loss"] for trial in pickle.load(open(save_loc, "rb")).results]
        os.remove(save_loc)
        results.append(vals)
    else:
        with open(path, mode="w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    for bench in FUNCS:
        if isinstance(bench, ABCMeta):
            for d in [5, 10, 30]:
                collect_data(bench, dim=d)
        else:
            collect_data(bench)
