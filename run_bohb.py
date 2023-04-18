import os

import numpy as np

from baseline_utils._bohb import run_bohb
from baseline_utils.utils import get_subdir_name, parse_args
from tpe.utils.tabular_benchmarks import HPOLib, JAHSBench201, LCBench


BENCH_CHOICES = dict(lc=LCBench, hpolib=HPOLib, jahs=JAHSBench201)


if __name__ == "__main__":
    args = parse_args()
    subdir_name = get_subdir_name(args)
    np.random.seed(args.seed)
    obj_func = BENCH_CHOICES[args.bench_name](dataset_id=args.dataset_id, seed=args.seed)

    run_id = f"bench={args.bench_name}_dataset={args.dataset_id}_nworkers={args.n_workers}_seed={args.seed}"
    run_bohb(
        obj_func=obj_func,
        config_space=obj_func.config_space,
        min_budget=obj_func.min_budget,
        max_budget=obj_func.max_budget,
        run_id=run_id,
        n_workers=args.n_workers,
        subdir_name=os.path.join("bohb", subdir_name),
    )
