import itertools
import os

from tpe.optimizer import TPEOptimizer
from tpe.utils.tabular_benchmarks import AbstractBench, HPOLib
from tpe.utils.constants import QuantileFunc

from experiment_utils import exist_file, save_observations


FUNCS = [HPOLib(dataset_id=i, seed=None) for i in range(4)]
N_SEEDS = 10
MAX_EVALS = 200
N_INIT = 200 * 5 // 100
LINEAR, SQRT = "linear", "sqrt"


def collect_data(
    bench: AbstractBench,
    multivariate: bool,
    choice: str,
    alpha: float,
    weight_func_choice: str,
    magic_clip: bool,
) -> None:

    func_name = bench.dataset_name
    print(func_name, multivariate, choice, alpha, weight_func_choice)
    dir_name = "_".join(
        [
            f"multivariate={multivariate}",
            f"quantile={choice}",
            f"alpha={alpha}",
            f"weight={weight_func_choice}",
            f"magic-clip={magic_clip}",
        ]
    )
    file_name = f"{func_name}.json"
    if exist_file(dir_name, file_name):
        return

    results = []
    for seed in range(N_SEEDS):
        bench.reseed(seed)
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
            magic_clip=magic_clip,
            top=None,
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
                magic_clip=params[3],
            )
        except Exception as e:
            print(f"Failed with error {e}")
