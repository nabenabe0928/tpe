from argparse import ArgumentParser

import itertools
import os

import numpy as np

from tpe.optimizer import TPEOptimizer
from tpe.utils.tabular_benchmarks import JAHSBench201
from tpe.utils.constants import QuantileFunc

from experiment_utils import exist_file, save_observations


parser = ArgumentParser()
parser.add_argument("--dataset_id", type=int, choices=list(range(3)))
args = parser.parse_args()

# FUNCS = [JAHSBench201(dataset_id=i) for i in range(3)]
FUNCS = [JAHSBench201(dataset_id=args.dataset_id)]
N_SEEDS = 10
MAX_EVALS = 200
N_INIT = 200 * 5 // 100
LINEAR, SQRT = "linear", "sqrt"


def collect_data(
    bench: JAHSBench201,
    choice: str,
    alpha: float,
    weight_func_choice: str,
    magic_clip_exponent: float,
    heuristic: bool,
    min_bandwidth_factor: float,
) -> None:

    func_name = bench.dataset_name
    heuristic_name = heuristic if heuristic is not None else "scott"
    print(func_name, choice, alpha, weight_func_choice, heuristic_name, min_bandwidth_factor)
    magic_clip_exponent_str = str(magic_clip_exponent) if magic_clip_exponent != np.inf else "inf"
    dir_name = "_".join(
        [
            f"quantile={choice}",
            f"alpha={alpha}",
            f"weight={weight_func_choice}",
            f"magic-clip-exponent={magic_clip_exponent_str}",
            f"heuristic={heuristic_name}",
            f"min_bandwidth_factor={min_bandwidth_factor}",
        ]
    )
    file_name = f"{func_name}.json"
    if exist_file(dir_name, file_name, result_dir="results-bandwidth/"):
        return

    results = []
    print("Collect", dir_name, file_name)
    for seed in range(N_SEEDS):
        opt = TPEOptimizer(
            obj_func=bench,
            config_space=bench.config_space,
            max_evals=MAX_EVALS,
            n_init=N_INIT,
            weight_func_choice=weight_func_choice,
            quantile_func=QuantileFunc(choice=choice, alpha=alpha),
            seed=seed,
            resultfile=os.path.join(dir_name, file_name),
            magic_clip=magic_clip_exponent != np.inf,
            magic_clip_exponent=magic_clip_exponent,
            top=None,
            heuristic=heuristic,
            min_bandwidth_factor=min_bandwidth_factor,
        )
        opt.optimize()
        results.append(opt.fetch_observations()["loss"].tolist())

    save_observations(dir_name=dir_name, file_name=file_name, data=results, result_dir="results-bandwidth/")


if __name__ == "__main__":
    for params in itertools.product(
        *(
            [
                (LINEAR, 0.05),
                (LINEAR, 0.10),
                (LINEAR, 0.15),
                (LINEAR, 0.20),
                (SQRT, 0.25),
                (SQRT, 0.5),
                (SQRT, 0.75),
                (SQRT, 1.0),
            ],  # quantile_func
            ["uniform", "older-smaller", "expected-improvement"],  # weight_func_choice
            [0.25, 0.5, 1.0, 2.0, 4.0, np.inf],  # magic_clip
            [None, "optuna", "hyperopt"],  # heuristics
            [0.01, 0.03, 0.1, 0.3],  # min bandwidth factor
            FUNCS,
        )
    ):
        try:
            collect_data(
                bench=params[-1],
                choice=params[0][0],
                alpha=params[0][1],
                weight_func_choice=params[1],
                magic_clip_exponent=params[2],
                heuristic=params[3],
                min_bandwidth_factor=params[4],
            )
        except Exception as e:
            print(f"Failed with error {e}")
