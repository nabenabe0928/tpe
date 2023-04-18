import os

import numpy as np

from baseline_utils._dehb import run_dehb
from baseline_utils.utils import get_subdir_name, parse_args
from tpe.utils.tabular_benchmarks import HPOLib, JAHSBench201, LCBench


BENCH_CHOICES = dict(lc=LCBench, hpolib=HPOLib, jahs=JAHSBench201)


class Wrapper:
    def __init__(self, bench):
        self._bench = bench

    def __call__(self, eval_config, budget):
        output = self._bench(eval_config, budget)
        ret_vals = dict(fitness=output["loss"], cost=output["runtime"])
        return ret_vals


if __name__ == "__main__":
    args = parse_args()
    subdir_name = get_subdir_name(args)
    np.random.seed(args.seed)
    bench = BENCH_CHOICES[args.bench_name](dataset_id=args.dataset_id, seed=args.seed)
    wrapped_func = Wrapper(bench)

    run_dehb(
        obj_func=wrapped_func,
        config_space=bench.config_space,
        min_budget=bench.min_budget,
        max_budget=bench.max_budget,
        n_workers=args.n_workers,
        subdir_name=os.path.join("dehb", subdir_name),
    )
