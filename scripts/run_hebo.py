# use HEBO: https://github.com/huawei-noah/HEBO/tree/master/HEBO

import json
import os
import time
from abc import ABCMeta
from typing import Callable, List, Optional, Tuple

import ConfigSpace as CS

import numpy as np

import pandas as pd

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

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
FUNCS += [LCBench(dataset_id=i, seed=None) for i in range(34)]
FUNCS += [NASHPOBench2(dataset_id=0, seed=None)]
FUNCS += [OlympusBench(dataset_id=i, seed=None) for i in range(10)]


def wrapper_func(bench: Callable) -> Callable:
    def func(configs: pd.DataFrame) -> np.ndarray:
        eval_config = {hp_name: configs[hp_name].iloc[0] for hp_name in configs.columns}
        target = bench.func if isinstance(bench, ABCMeta) else bench
        return np.asarray([[target(eval_config)]])

    return func


def get_init_configs(
    bench: Callable,
    config_space: CS.ConfigurationSpace,
    seed: int,
) -> Tuple[pd.DataFrame, List[float]]:
    random_sampler = TPEOptimizer(
        obj_func=bench.func if isinstance(bench, ABCMeta) else bench,
        config_space=config_space,
        n_init=10,
        max_evals=10,
        seed=seed,
    )
    random_sampler.optimize()
    init_data = random_sampler.fetch_observations()
    vals = init_data["loss"]
    init_configs = pd.DataFrame([{
            hp_name: init_data[hp_name][i] for hp_name in config_space
        } for i in range(vals.size)
    ])
    return init_configs, list(vals)


def extract_space(config_space: CS.ConfigurationSpace):
    config_info = []
    for hp in config_space.get_hyperparameters():
        info = {"name": hp.name}
        if isinstance(hp, CS.CategoricalHyperparameter):
            info["type"] = "cat"
            info["categories"] = hp.choices
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            log = hp.log
            info["type"] = "pow" if log else "num"
            info["lb"], info["ub"] = hp.lower, hp.upper
            if log:
                info["base"] = 10
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            info["type"] = "int"
            info["lb"], info["ub"] = hp.lower, hp.upper
        else:
            raise TypeError(f"{type(type(hp))} is not supported")

        config_info.append(info)

    return DesignSpace().parse(config_info)


def collect_data(bench: Callable, dim: Optional[int] = None) -> None:
    bench_name = bench().__class__.__name__ if isinstance(bench, ABCMeta) else bench.dataset_name
    bench_name = f"{bench_name}_{dim:0>2}d" if dim is not None else bench_name
    print(bench_name, dim)

    config_space = CS.ConfigurationSpace()
    if dim is not None:
        config_space.add_hyperparameters(
            [CS.UniformFloatHyperparameter(f"x{d}", lower=-1, upper=1) for d in range(dim)]
        )

    dir_name = "results/hebo"
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
        print(f"Start {bench_name} {dim} with seed {seed} at {time.time()}")
        if hasattr(bench, "reseed"):
            bench.reseed(seed)

        config_space = config_space if isinstance(bench, ABCMeta) else bench.config_space
        init_configs, vals = get_init_configs(bench, config_space, seed)
        obj = wrapper_func(bench)

        opt = HEBO(extract_space(config_space), rand_sample=10)
        opt.observe(init_configs, np.asarray(vals).reshape(-1, 1))
        for i in range(190):
            if (i + 1) % 10 == 0:
                print(f"{i + 11} evaluations at {time.time()}")

            config = opt.suggest(n_suggestions=1)
            y = obj(config)
            opt.observe(config, y)
            vals.append(float(y[0][0]))

        print(vals)
        results.append(vals)
    else:
        with open(path, mode="w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    for bench in FUNCS:
        try:
            if isinstance(bench, ABCMeta):
                for d in [5, 10, 30]:
                    collect_data(bench, dim=d)
            else:
                collect_data(bench)
        except Exception as e:
            print(f"Failed with error {e}")
