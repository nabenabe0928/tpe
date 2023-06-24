# Check: https://github.com/awslabs/syne-tune/blob/main/syne_tune/optimizer/schedulers/searchers/bore/bore.py

import json
import os
import time
from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ConfigSpace as CS

import numpy as np

import pandas as pd

import torch

import xgboost

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
from tpe.utils.tabular_benchmarks import HPOBench, HPOLib, JAHSBench201, LCBench, NASHPOBench2


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


def wrapper_func(bench: Callable) -> Callable:
    def func(eval_config: Dict[str, Any]) -> float:
        target = bench.func if isinstance(bench, ABCMeta) else bench
        return target(eval_config)

    return func


def get_pandas_value(
    config_space: CS.ConfigurationSpace, hp_name: str, values: np.ndarray
) -> Union[pd.Categorical, np.ndarray]:
    hp = config_space.get_hyperparameter(hp_name)
    if isinstance(hp, CS.CategoricalHyperparameter):
        return pd.Categorical(values, categories=hp.choices)
    else:
        return values


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
    init_configs = pd.DataFrame({
        hp_name: get_pandas_value(config_space, hp_name, init_data[hp_name])
        for hp_name in config_space
    })
    return init_configs, list(vals)


def random_sample(config: CS.hyperparameters, rng: np.random.RandomState) -> np.ndarray:
    n_samples = 500  # the default from the original paper
    if isinstance(config, CS.CategoricalHyperparameter):
        return pd.Categorical(rng.choice(config.choices, size=n_samples), categories=config.choices)
    elif isinstance(config, CS.UniformFloatHyperparameter):
        lb = np.log(config.lower) if config.log else config.lower
        ub = np.log(config.upper) if config.log else config.upper
        rnd = rng.random(n_samples) * (ub - lb) + lb
        return np.exp(rnd) if config.log else rnd
    elif isinstance(config, CS.UniformIntegerHyperparameter):
        r = config.upper - config.lower
        return rng.randint(r, size=n_samples) + config.lower
    else:
        raise TypeError(f"{type(type(config))} is not supported")


def ask(
    config_space: CS.ConfigurationSpace,
    configs: pd.DataFrame,
    vals: List[float],
    rng: np.random.RandomState,
    seed: int,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemError(
            "Due to the version confliction between JAHS-Bench-201 and tree_method in xgboost, "
            "we cannot run BORE without CUDA."
        )

    # valid tree_method --> gpu_hist (from 1.5), approx (from 1.6), hist (from 1.7)
    # JAHS-Bench-201 works only on xgboost==1.5, so we need to choose gpu_hist.
    model = xgboost.XGBClassifier(seed=seed, eval_metric="logloss", tree_method="gpu_hist", enable_categorical=True)
    threshold = np.quantile(vals, q=0.25)
    z = np.less(vals, threshold)

    model.fit(configs, z.astype(np.int64))
    candidates = pd.DataFrame({
        hp_name: random_sample(config=config_space.get_hyperparameter(hp_name), rng=rng)
        for hp_name in config_space
    })
    probas = model.predict_proba(candidates)[:, 1]  # proba of the class 1
    return candidates.iloc[np.argmax(probas)].to_dict()


def collect_data(bench: Callable, dim: Optional[int] = None) -> None:
    bench_name = bench().__class__.__name__ if isinstance(bench, ABCMeta) else bench.dataset_name
    bench_name = f"{bench_name}_{dim:0>2}d" if dim is not None else bench_name
    print(bench_name, dim)

    config_space = CS.ConfigurationSpace()
    if dim is not None:
        config_space.add_hyperparameters(
            [CS.UniformFloatHyperparameter(f"x{d}", lower=-1, upper=1) for d in range(dim)]
        )

    dir_name = "results/bore"
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
        configs, vals = get_init_configs(bench, config_space, seed)
        obj = wrapper_func(bench)
        rng = np.random.RandomState(seed)

        for i in range(190):
            config = ask(config_space, configs, vals, rng, seed=seed)
            y = float(obj(config))
            new_config = pd.DataFrame({
                hp_name: get_pandas_value(config_space, hp_name, np.asarray([val]))
                for hp_name, val in config.items()
            }, index=[len(configs)])
            configs = pd.concat([configs, new_config])
            vals.append(y)

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
