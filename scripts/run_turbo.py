import json
import os
import time
from abc import ABCMeta
from typing import Any, Callable, Dict, Optional, Union

import ConfigSpace as CS

import numpy as np

from turbo.turbo import Turbo1

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
FUNCS += [JAHSBench201(dataset_id=i) for i in range(2, 3)]
FUNCS += [LCBench(dataset_id=i) for i in range(8)]
FUNCS += [NASHPOBench2(dataset_id=0)]
FUNCS += [OlympusBench(dataset_id=i) for i in range(10)]


def get_bounds(
    config_space: CS.ConfigurationSpace
) -> Union[np.ndarray, np.ndarray]:
    hp_names = []
    lb, ub = [], []
    for hp_name in config_space:
        hp_names.append(hp_name)
        hp = config_space.get_hyperparameter(hp_name)
        if isinstance(hp, CS.CategoricalHyperparameter):
            for i in range(len(hp.choices)):
                lb.append(0)
                ub.append(1)
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            lb.append(np.log(hp.lower) if hp.log else hp.lower)
            ub.append(np.log(hp.upper) if hp.log else hp.upper)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            lb.append(hp.lower - 0.5 + 1e-12)
            ub.append(hp.upper + 0.5 - 1e-12)

    return np.asarray(lb), np.asarray(ub)


def convert(
    X: np.ndarray,
    config_space: CS.ConfigurationSpace,
) -> Dict[str, Any]:
    config = {}
    cur = 0
    for hp_name in config_space:
        hp = config_space.get_hyperparameter(hp_name)
        if isinstance(hp, CS.UniformFloatHyperparameter):
            config[hp_name] = np.exp(X[cur]) if hp.log else X[cur]
            cur += 1
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            config[hp_name] = int(np.round(X[cur]))
            cur += 1
        elif isinstance(hp, CS.CategoricalHyperparameter):
            config[hp_name] = int(np.argmax([X[cur + i] for i in range(len(hp.choices))]))
            cur += len(hp.choices)

    return config


def wrapper_func(bench: Callable, config_space: CS.ConfigurationSpace) -> Callable:
    def func(X: np.ndarray) -> float:
        target = bench.func if isinstance(bench, ABCMeta) else bench
        eval_config = convert(X, config_space)
        return target(eval_config)

    return func


def get_init_config(
    bench: Callable,
    config_space: CS.ConfigurationSpace,
    seed: int,
) -> np.ndarray:
    random_sampler = TPEOptimizer(
        obj_func=bench.func if isinstance(bench, ABCMeta) else bench,
        config_space=config_space,
        n_init=10,
        max_evals=10,
        seed=seed,
    )
    random_sampler.optimize()
    init_data = random_sampler.fetch_observations()
    init_X = []
    for hp_name in config_space:
        hp = config_space.get_hyperparameter(hp_name)
        if isinstance(hp, CS.CategoricalHyperparameter):
            init_choices = [int(c) for c in init_data[hp_name]]
            assert len(init_choices) == 10
            one_hot = np.zeros((len(hp.choices), 10))
            for idx, c in enumerate(init_choices):
                one_hot[c, idx] = 1.0

            init_X.extend(one_hot.tolist())
        else:
            init_X.append(np.log(init_data[hp_name]) if hp.log else init_data[hp_name])

    return np.asarray(init_X).T, np.asarray(init_data["loss"])


def collect_data(bench: Callable, dim: Optional[int] = None) -> None:
    bench_name = bench().__class__.__name__ if isinstance(bench, ABCMeta) else bench.dataset_name
    bench_name = f"{bench_name}_{dim:0>2}d" if dim is not None else bench_name
    print(bench_name, dim)

    config_space = CS.ConfigurationSpace()
    if dim is not None:
        config_space.add_hyperparameters(
            [CS.UniformFloatHyperparameter(f"x{d}", lower=-1, upper=1) for d in range(dim)]
        )
    dir_name = "results/turbo"
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
        X_init, fX_init = get_init_config(bench, config_space, seed)
        lb, ub = get_bounds(config_space)
        opt = Turbo1(
            f=wrapper_func(bench, config_space),
            lb=lb,
            ub=ub,
            n_init=10,
            max_evals=200,
            seed=seed,
            verbose=False,
        )
        opt.optimize(fixed_X_init=X_init, fixed_fX_init=fX_init)
        results.append(opt.fX.flatten().tolist())
    else:
        with open(path, mode="w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    for bench in FUNCS:
        if isinstance(bench, ABCMeta):
            for d in [30]:
                collect_data(bench, dim=d)
        else:
            collect_data(bench)
