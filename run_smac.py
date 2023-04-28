import os

import numpy as np

from baseline_utils._smac import run_smac
from baseline_utils.utils import get_subdir_name, parse_args
from tpe.utils.tabular_benchmarks import HPOLib, JAHSBench201, LCBench


BENCH_CHOICES = dict(lc=LCBench, hpolib=HPOLib, jahs=JAHSBench201)


class Wrapper:
    def __init__(self, bench):
        self._bench = bench

    def get_shared_data(self):
        kwargs = dict(shared_data=self._bench.get_data()) if hasattr(self._bench, "get_data") else {}
        return kwargs

    def __call__(self, eval_config, budget, bench_data=None):
        return self._bench(eval_config, budget, **({} if bench_data is None else dict(bench_data=bench_data)))


if __name__ == "__main__":
    args = parse_args()
    subdir_name = get_subdir_name(args)
    np.random.seed(args.seed)
    obj_func = BENCH_CHOICES[args.bench_name](dataset_id=args.dataset_id, seed=args.seed, keep_benchdata=False)
    wrapper = Wrapper(obj_func)

    run_smac(
        obj_func=wrapper,
        config_space=obj_func.config_space,
        min_budget=obj_func.min_budget,
        max_budget=obj_func.max_budget,
        n_workers=args.n_workers,
        subdir_name=os.path.join("smac", subdir_name),
        n_init=5,  # TODO: check
    )
