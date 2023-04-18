import os

import ConfigSpace as CS

import numpy as np

import neps

from baseline_utils._neps import run_neps
from baseline_utils.utils import get_subdir_name, parse_args
from tpe.utils.tabular_benchmarks import HPOLib, JAHSBench201, LCBench


BENCH_CHOICES = dict(lc=LCBench, hpolib=HPOLib, jahs=JAHSBench201)
BUDGET_KEY = "epoch"


class Wrapper:
    def __init__(self, bench):
        self._bench = bench

    def __call__(self, eval_config, budget):
        output = self._bench(eval_config, budget)
        ret_vals = dict(loss=output["loss"], cost=output["runtime"])
        return ret_vals


def get_pipeline_space(config_space: CS.ConfigurationSpace):
    pipeline_space = {}
    for hp_name in bench.config_space:
        hp = bench.config_space.get_hyperparameter(hp_name)
        if isinstance(hp, CS.UniformFloatHyperparameter):
            pipeline_space[hp.name] = neps.FloatParameter(lower=hp.lower, upper=hp.upper, log=hp.log)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            pipeline_space[hp.name] = neps.IntegerParameter(lower=hp.lower, upper=hp.upper, log=hp.log)
        elif isinstance(hp, CS.CategoricalHyperparameter):
            pipeline_space[hp.name] = neps.CategoricalParameter(choices=hp.choices)
        else:
            raise ValueError(f"{type(hp)} is not supported")

    return pipeline_space


if __name__ == "__main__":
    args = parse_args()
    subdir_name = get_subdir_name(args)
    np.random.seed(args.seed)
    bench = BENCH_CHOICES[args.bench_name](dataset_id=args.dataset_id, seed=args.seed)
    wrapped_func = Wrapper(bench)

    min_budget, max_budget = bench.min_budget, bench.max_budget
    pipeline_space = get_pipeline_space(bench.config_space)
    pipeline_space[BUDGET_KEY] = neps.IntegerParameter(lower=min_budget, upper=max_budget, is_fidelity=True)

    run_neps(
        obj_func=wrapped_func,
        pipeline_space=pipeline_space,
        min_budget=min_budget,
        max_budget=max_budget,
        n_workers=args.n_workers,
        subdir_name=os.path.join("neps", subdir_name),
        budget_key=BUDGET_KEY,
        max_evals=22,
    )
