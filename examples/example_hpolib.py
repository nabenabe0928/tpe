"""
Please install the dependencies first:

```shell
$ pip install optuna==4.0.0b0 optunahub
```

"""

from argparse import ArgumentParser
import pickle

import numpy as np

import optuna

import optunahub


parser = ArgumentParser()
parser.add_argument("--dataset_id", choices=list(range(4)), type=int, required=True)
args = parser.parse_args()
dataset_id = args.dataset_id
dataset_names = ["naval_propulsion", "parkinsons_telemonitoring", "protein_structure", "slice_localization"]
dataset_name = dataset_names[dataset_id]
dataset = pickle.load(open(f"examples/{dataset_name}.pkl", mode="rb"))
seed = 0
rng = np.random.RandomState(seed)


def objective(trial: optuna.Trial) -> float:
    # These are the indices of each hyperparameter.
    hyperparameter_indices = [
        trial.suggest_categorical("activation_fn_1", list(range(2))),
        trial.suggest_categorical("activation_fn_2", list(range(2))),
        trial.suggest_int("batch_size", low=0, high=3),
        trial.suggest_int("dropout_1", low=0, high=2),
        trial.suggest_int("dropout_2", low=0, high=2),
        trial.suggest_int("init_lr", low=0, high=5),
        trial.suggest_categorical("lr_schedule", list(range(2))),
        trial.suggest_int("n_units_1", low=0, high=5),
        trial.suggest_int("n_units_2", low=0, high=5),
    ]
    config_id = "".join([str(i) for i in hyperparameter_indices])
    eval_seed = rng.randint(4)
    return dataset[config_id][eval_seed]


module = optunahub.load_module(package="samplers/tpe_tutorial")
tpe_config = {
    "consider_prior": True,
    "consider_magic_clip": True,
    "multivariate": True,
    "b_magic_exponent": 1.0,
    "min_bandwidth_factor": 0.01,
    "gamma_strategy": "linear",
    "gamma_beta": 0.1,
    "weight_strategy": "old-decay",
    "bandwidth_strategy": "hyperopt",
    "categorical_prior_weight": None,
}

sampler = module.CustomizableTPESampler(seed=seed)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
