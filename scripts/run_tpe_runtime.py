from argparse import ArgumentParser

import ConfigSpace as CS

from tpe.optimizer import TPEOptimizer
from tpe.utils.benchmarks import Sphere
from tpe.utils.constants import QuantileFunc


parser = ArgumentParser()
parser.add_argument("--dim", type=int, choices=[5, 10, 30], required=True)
args = parser.parse_args()
DIM = args.dim
N_SEEDS = 10
MAX_EVALS = 200
N_INIT = 200 * 5 // 100
LINEAR, SQRT = "linear", "sqrt"
CONFIG_SPACE = CS.ConfigurationSpace()
CONFIG_SPACE.add_hyperparameters([CS.UniformFloatHyperparameter(name=f"x{d}", lower=-1, upper=1) for d in range(DIM)])


if __name__ == "__main__":
    heuristic_name = "hyperopt"

    results = []
    for seed in range(N_SEEDS):
        opt = TPEOptimizer(
            obj_func=Sphere.func,
            config_space=CONFIG_SPACE,
            max_evals=MAX_EVALS,
            n_init=N_INIT,
            weight_func_choice="expected-improvement",
            quantile_func=QuantileFunc(choice=LINEAR, alpha=0.15),
            seed=seed,
            resultfile="temp",
            magic_clip=True,
            magic_clip_exponent=2.0,
            heuristic="hyperopt",
            min_bandwidth_factor=0.03,
        )
        opt.optimize()
        results.append(opt.fetch_observations()["loss"].tolist())
