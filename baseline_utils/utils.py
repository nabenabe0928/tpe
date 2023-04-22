from argparse import ArgumentParser, Namespace

from tpe.utils.tabular_benchmarks import HPOLib, JAHSBench201, LCBench


BENCH_CHOICES = dict(lc=LCBench, hpolib=HPOLib, jahs=JAHSBench201)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_id", type=int, choices=list(range(34)))
    parser.add_argument("--bench_name", type=str, choices=list(BENCH_CHOICES.keys()))
    parser.add_argument("--n_workers", type=int)
    args = parser.parse_args()
    return args


def get_subdir_name(args: Namespace) -> str:
    dataset = BENCH_CHOICES[args.bench_name]._DATASET_NAMES[args.dataset_id]
    return f"bench={args.bench_name}_dataset={dataset}_nworkers={args.n_workers}/{args.seed}"
