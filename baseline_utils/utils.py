from argparse import ArgumentParser, Namespace

import ConfigSpace as CS

from tpe.utils.tabular_benchmarks import HPOLib, JAHSBench201, LCBench


BENCH_CHOICES = dict(lc=LCBench, hpolib=HPOLib, jahs=JAHSBench201)


class TestFunc:
    def __call__(self, eval_config, budget):
        return dict(loss=eval_config["x"]**2, runtime=budget / 1.0)

    @property
    def config_space(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5.0, 5.0))
        return cs

    @property
    def min_budget(self):
        return 1

    @property
    def max_budget(self):
        return 9


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
