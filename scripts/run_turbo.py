import json
import os
import shutil
from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Tuple

import ConfigSpace as CS

import numpy as np

from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.optimizer.configuration_chooser import TurBOChooser

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
from tpe.utils.tabular_benchmarks import HPOBench, HPOLib, JAHSBench201


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


def wrapper_func(bench: Callable) -> Callable:
    def func(config: Dict[str, Any]) -> float:
        target = bench.func if isinstance(bench, ABCMeta) else bench
        return target(config)

    return func


def update_config_space_default_and_get_init_config(
    bench: Callable,
    config_space: CS.ConfigurationSpace,
    seed: int,
) -> Tuple[CS.ConfigurationSpace, List[CS.Configuration]]:
    random_sampler = TPEOptimizer(
        obj_func=bench.func if isinstance(bench, ABCMeta) else bench,
        config_space=config_space,
        n_init=10,
        max_evals=10,
        seed=seed,
    )
    random_sampler.optimize()
    data = random_sampler.fetch_observations()
    for hp in config_space.get_hyperparameters():
        hp.default_value = data[hp.name][-1]

    n_init = data["loss"].size
    assert n_init == 10
    init_configs = [
        CS.Configuration(
            config_space,
            values={
                hp_name: data[hp_name][i] for hp_name in config_space
            })
        for i in range(n_init - 1)
    ]
    return config_space, init_configs


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
        if hasattr(bench, "reseed"):
            bench.reseed(seed)

        config_space = config_space if isinstance(bench, ABCMeta) else bench.config_space
        config_space, init_configs = update_config_space_default_and_get_init_config(bench, config_space, seed)
        scenario = Scenario({
            "run_obj": "quality",
            "runcount-limit": 200,
            "cs": config_space,
        })
        if hasattr(bench, "reseed"):
            # We need to reseed again because SMAC doubly evaluates the init configs
            bench.reseed(seed)
        opt = SMAC4HPO(
            scenario=scenario,
            rng=np.random.RandomState(seed),
            tae_runner=wrapper_func(bench),
            model_type="gp",
            smbo_kwargs={"epm_chooser": TurBOChooser},
            initial_configurations=init_configs,
            initial_design=None,
        )
        opt.optimize()
        vals = [v.cost for v in opt.runhistory.data.values()]
        results.append(vals)

        for f in os.listdir():
            if f.startswith("smac3-output"):
                shutil.rmtree(f)
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
