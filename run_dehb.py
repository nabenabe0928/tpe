import os

import numpy as np

from baseline_utils._dehb import run_dehb
from baseline_utils.utils import get_subdir_name, parse_args
from tpe.utils.tabular_benchmarks import HPOLib, JAHSBench201, LCBench


BENCH_CHOICES = dict(lc=LCBench, hpolib=HPOLib, jahs=JAHSBench201)


class Wrapper:
    def __init__(self, bench):
        self._bench = bench

    def get_shared_data(self):
        kwargs = dict(bench_data=self._bench.get_data()) if hasattr(self._bench, "get_data") else {}
        return kwargs

    def __call__(self, eval_config, budget, bench_data=None):
        output = self._bench(eval_config, budget, **({} if bench_data is None else dict(bench_data=bench_data)))
        ret_vals = dict(fitness=output["loss"], cost=output["runtime"])
        return ret_vals


if __name__ == "__main__":
    args = parse_args()
    subdir_name = get_subdir_name(args)
    np.random.seed(args.seed)
    bench = BENCH_CHOICES[args.bench_name](dataset_id=args.dataset_id, seed=args.seed, keep_benchdata=False)
    wrapped_func = Wrapper(bench)

    run_dehb(
        obj_func=wrapped_func,
        config_space=bench.config_space,
        min_budget=bench.min_budget,
        max_budget=bench.max_budget,
        n_workers=args.n_workers,
        subdir_name=os.path.join("dehb", subdir_name),
    )
