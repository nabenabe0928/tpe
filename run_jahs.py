import itertools
import json
import os
from typing import Dict, List

from tpe.optimizer import TPEOptimizer
from tpe.utils.tabular_benchmarks import JAHSBench201
from tpe.utils.constants import QuantileFunc


FUNCS = [JAHSBench201(dataset_id=i) for i in range(3)]
N_SEEDS = 10
MAX_EVALS = 200
N_INIT = 200 * 5 // 100
LINEAR, SQRT = "linear", "sqrt"


def save_observations(
    dir_name: str,
    file_name: str,
    data: Dict[str, List[List[float]]]
) -> None:
    path = os.path.join("results", dir_name)
    with open(os.path.join(path, file_name), mode="w") as f:
        json.dump(data, f, indent=4)


def exist_file(dir_name: str, file_name: str) -> bool:
    path = os.path.join("results", dir_name)
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
    bench: JAHSBench201,
    min_bandwidth_factor: float,
    min_bandwidth_factor_for_discrete: float,
    multivariate: bool,
    choice: str,
    alpha: float,
    weight_func_choice: str,
    top: float,
) -> None:

    func_name = bench.dataset_name
    dir_name = "_".join([
        f"multivariate={multivariate}",
        f"quantile={choice}",
        f"alpha={alpha}",
        f"weight={weight_func_choice}",
        f"min_bandwidth_factor={min_bandwidth_factor}",
        f"min_bandwidth_factor_for_discrete={min_bandwidth_factor_for_discrete}",
        f"top={top}",
    ])
    file_name = f"{func_name}.json"
    if exist_file(dir_name, file_name):
        return

    results = []
    print("Collect", dir_name, file_name)
    top = top if top < 1.1 else None
    for seed in range(N_SEEDS):
        opt = TPEOptimizer(
            obj_func=bench,
            config_space=bench.config_space,
            max_evals=MAX_EVALS,
            n_init=N_INIT,
            weight_func_choice=weight_func_choice,
            quantile_func=QuantileFunc(choice=choice, alpha=alpha),
            multivariate=multivariate,
            min_bandwidth_factor_for_discrete=min_bandwidth_factor_for_discrete,
            seed=seed,
            resultfile=os.path.join(dir_name, file_name),
            top=top,
        )
        opt.optimize()
        results.append(opt.fetch_observations()["loss"].tolist())

    save_observations(dir_name=dir_name, file_name=file_name, data=results)


if __name__ == "__main__":
    for params in itertools.product(*(
        [0.01, 0.03, 0.1, 0.3],  # min_bandwidth_factor
        [0.25, 0.5, 1.0, 2.0],  # min_bandwidth_factor_for_discrete
        [True, False],  # multivariate
        [
            (LINEAR, 0.05), (LINEAR, 0.1), (LINEAR, 0.15), (LINEAR, 0.2),
            (SQRT, 0.25), (SQRT, 0.5), (SQRT, 0.75), (SQRT, 1.0),
        ],  # quantile_func
        ["uniform", "older-smaller", "expected-improvement", "weaker-smaller"],  # weight_func_choice
        [0.8, 0.9, 1.0, 2.0],  # top/ 2.0 is for the Optuna version
        FUNCS,
    )):
        try:
            collect_data(
                bench=params[-1],
                min_bandwidth_factor=params[0],
                min_bandwidth_factor_for_discrete=params[1],
                multivariate=params[2],
                choice=params[3][0],
                alpha=params[3][1],
                weight_func_choice=params[4],
                top=params[5],
            )
        except Exception as e:
            print(f"Failed with error {e}")
