from argparse import ArgumentParser

import itertools
import os

from tpe.optimizer import TPEOptimizer
from tpe.utils.tabular_benchmarks import JAHSBench201
from tpe.utils.constants import QuantileFunc

from experiment_utils import exist_file, save_observations


parser = ArgumentParser()
parser.add_argument("dataset_id", type=int, choices=list(range(3)))
args = parser.parse_args()

# FUNCS = [JAHSBench201(dataset_id=i) for i in range(3)]
FUNCS = [JAHSBench201(dataset_id=args.dataset_id)]
N_SEEDS = 10
MAX_EVALS = 200
N_INIT = 200 * 5 // 100
LINEAR, SQRT = "linear", "sqrt"


def collect_data(
    bench: JAHSBench201,
    multivariate: bool,
    choice: str,
    alpha: float,
    weight_func_choice: str,
    prior: bool,
    magic_clip: bool,
    top: float,
) -> None:

    func_name = bench.dataset_name
    dir_name = "_".join(
        [
            f"multivariate={multivariate}",
            f"quantile={choice}",
            f"alpha={alpha}",
            f"weight={weight_func_choice}",
            f"prior={prior}",
            f"magic-clip={magic_clip}"
            f"top={top}",
        ]
    )
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
            seed=seed,
            resultfile=os.path.join(dir_name, file_name),
            prior=prior,
            magic_clip=magic_clip,
            top=top,
        )
        opt.optimize()
        results.append(opt.fetch_observations()["loss"].tolist())

    save_observations(dir_name=dir_name, file_name=file_name, data=results)


if __name__ == "__main__":
    for params in itertools.product(
        *(
            [True, False],  # multivariate
            [
                (LINEAR, 0.05),
                (LINEAR, 0.1),
                (LINEAR, 0.15),
                (LINEAR, 0.2),
                (SQRT, 0.25),
                (SQRT, 0.5),
                (SQRT, 0.75),
                (SQRT, 1.0),
            ],  # quantile_func
            ["uniform", "older-smaller", "older-drop", "expected-improvement", "weaker-smaller"],  # weight_func_choice
            [True, False],  # prior
            [True, False],  # magic_clip
            [0.8, 0.9, 1.0, 2.0],  # top/ 2.0 is for the Optuna version
            FUNCS,
        )
    ):
        try:
            collect_data(
                bench=params[-1],
                multivariate=params[0],
                choice=params[1][0],
                alpha=params[1][1],
                weight_func_choice=params[2],
                prior=params[3],
                magic_clip=params[4],
                top=params[5],
            )
        except Exception as e:
            print(f"Failed with error {e}")
