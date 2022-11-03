import json
import os
from abc import ABCMeta
from typing import Callable, Optional

import ConfigSpace as CS

import optuna

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
)
from tpe.utils.tabular_benchmarks import HPOBench, HPOLib, JAHSBench201


optuna.logging.set_verbosity(optuna.logging.CRITICAL)
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
]
FUNCS += [HPOBench(dataset_id=i, seed=None) for i in range(8)]
FUNCS += [HPOLib(dataset_id=i, seed=None) for i in range(4)]
FUNCS += [JAHSBench201(dataset_id=i) for i in range(2)]


def wrapper_func(bench: Callable, config_space: CS.ConfigurationSpace) -> Callable:
    def func(trial: optuna.Trial) -> float:
        eval_config = {}
        for hp in config_space.get_hyperparameters():
            name = hp.name
            if isinstance(hp, CS.CategoricalHyperparameter):
                eval_config[name] = trial.suggest_categorical(name=name, choices=hp.choices)
            elif isinstance(hp, CS.UniformFloatHyperparameter):
                eval_config[name] = trial.suggest_float(name=name, low=hp.lower, high=hp.upper, log=hp.log)
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                eval_config[name] = trial.suggest_int(name=name, low=hp.lower, high=hp.upper)
            else:
                raise TypeError(f"{type(type(hp))} is not supported")

        target = bench.func if isinstance(bench, ABCMeta) else bench
        return target(eval_config)

    return func


def collect_data(bench: Callable, dim: Optional[int] = None) -> None:
    bench_name = bench().__class__.__name__ if isinstance(bench, ABCMeta) else bench.dataset_name
    bench_name = f"{bench_name}_{dim:0>2}d" if dim is not None else bench_name
    print(bench_name, dim)

    config_space = CS.ConfigurationSpace()
    if dim is not None:
        config_space.add_hyperparameters([
            CS.UniformFloatHyperparameter(f"x{d}", lower=-1, upper=1) for d in range(dim)
        ])

    for multivariate in [True, False]:
        print(f"multivariate: {multivariate}")
        mode = f"multivariate={multivariate}"
        dir_name = f"results/optuna_{mode}"
        os.makedirs(dir_name, exist_ok=True)
        path = os.path.join(dir_name, f"{bench_name}.json")
        if os.path.exists(path):
            print(f"{path} exists; skip")
            continue

        results = []
        for seed in range(10):
            if hasattr(bench, "reseed"):
                bench.reseed(seed)

            sampler = optuna.samplers.TPESampler(multivariate=multivariate, seed=seed, n_startup_trials=10)
            study = optuna.create_study(sampler=sampler)

            config_space = config_space if isinstance(bench, ABCMeta) else bench.config_space
            study.optimize(wrapper_func(bench=bench, config_space=config_space), n_trials=200)
            vals = [trial.value for trial in study.trials]
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
