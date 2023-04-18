import json
import os
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Union

import ConfigSpace as CS

import h5py

try:
    import jahs_bench
except ModuleNotFoundError:  # We cannot use jahs with smac
    pass

import numpy as np

from yahpo_gym import benchmark_set, local_config


DATA_DIR_NAME = os.path.join(os.environ["HOME"], "tabular_benchmarks")
VALUE_RANGES = json.load(open("tpe/utils/tabular_benchmarks.json"))
local_config.init_config()
local_config.set_data_path(DATA_DIR_NAME)


class AbstractBench(metaclass=ABCMeta):
    _rng: np.random.RandomState
    _value_range: Dict[str, List[Union[int, float, str]]]
    dataset_name: str

    def reseed(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    def _fetch_discrete_config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CS.UniformIntegerHyperparameter(name=name, lower=0, upper=len(choices) - 1)
                if not isinstance(choices[0], (str, bool))
                else CS.CategoricalHyperparameter(name=name, choices=[str(i) for i in range(len(choices))])
                for name, choices in self._value_range.items()
            ]
        )
        return config_space

    @property
    @abstractmethod
    def config_space(self) -> CS.ConfigurationSpace:
        raise NotImplementedError

    @property
    @abstractmethod
    def min_budget(self) -> int:
        # eta ** S <= R/r < eta ** (S + 1) to have S rungs.
        raise NotImplementedError

    @property
    @abstractmethod
    def max_budget(self) -> int:
        raise NotImplementedError


class LCBench(AbstractBench):
    # https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/
    _target_metric = "val_balanced_accuracy"
    _TRUE_MAX_BUDGET = 52

    def __init__(
        self,
        dataset_id: int,
        seed: Optional[int] = None,  # surrogate is not stochastic
    ):
        dataset_info = (
            ("kddcup09_appetency", "3945"),
            ("covertype", "7593"),
            ("amazon_employee_access", "34539"),
            ("adult", "126025"),
            ("nomao", "126026"),
            ("bank_marketing", "126029"),
            ("shuttle", "146212"),
            ("australian", "167104"),
            ("kr_vs_kp", "167149"),
            ("mfeat_factors", "167152"),
            ("credit_g", "167161"),
            ("vehicle", "167168"),
            ("kc1", "167181"),
            ("blood_transfusion_service_center", "167184"),
            ("cnae_9", "167185"),
            ("phoneme", "167190"),
            ("higgs", "167200"),
            ("connect_4", "167201"),
            ("helena", "168329"),
            ("jannis", "168330"),
            ("volkert", "168331"),
            ("mini_boo_ne", "168335"),
            ("aps_failure", "168868"),
            ("christine", "168908"),
            ("fabert", "168910"),
            ("airlines", "189354"),
            ("jasmine", "189862"),
            ("sylvine", "189865"),
            ("albert", "189866"),
            ("dionis", "189873"),
            ("car", "189905"),
            ("segment", "189906"),
            ("fashion_mnist", "189908"),
            ("jungle_chess_2pcs_raw_endgame_complete", "189909")
        )
        self.dataset_name, self._dataset_id = dataset_info[dataset_id]
        self._surrogate = benchmark_set.BenchmarkSet("lcbench", instance=self._dataset_id)
        self._config_space = self.config_space

    def __call__(self, config: Dict[str, Union[int, float]], budget: int = 52) -> Dict[str, float]:
        budget = int(min(self._TRUE_MAX_BUDGET, budget))
        EPS = 1e-12
        for hp in self._config_space.get_hyperparameters():
            lb, ub, name = hp.lower, hp.upper, hp.name
            if isinstance(hp, CS.UniformFloatHyperparameter):
                assert isinstance(config[name], float) and lb - EPS <= config[name] <= ub + EPS
            else:
                config[name] = int(config[name])
                assert isinstance(config[name], int) and lb <= config[name] <= ub

        config["OpenML_task_id"] = self._dataset_id
        config["epoch"] = budget
        output = self._surrogate.objective_function(config)[0]
        return dict(loss=1.0 - output[self._target_metric], runtime=output["time"])

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CS.UniformIntegerHyperparameter(name="batch_size", lower=16, upper=512, log=True),
                CS.UniformFloatHyperparameter(name="learning_rate", lower=1e-4, upper=0.1, log=True),
                CS.UniformFloatHyperparameter(name="max_dropout", lower=0.0, upper=1.0),
                CS.UniformIntegerHyperparameter(name="max_units", lower=64, upper=1024, log=True),
                CS.UniformFloatHyperparameter(name="momentum", lower=0.1, upper=0.9),
                CS.UniformIntegerHyperparameter(name="num_layers", lower=1, upper=5),
                CS.UniformFloatHyperparameter(name="weight_decay", lower=1e-5, upper=0.1),
            ]
        )
        return config_space

    @property
    def min_budget(self) -> int:
        return 6

    @property
    def max_budget(self) -> int:
        # in reality, the max_budget is 52, but make it 54 only for computational convenience.
        return 54


class HPOLib(AbstractBench):
    """
    Download the datasets via:
        $ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
        $ tar xf fcnet_tabular_benchmarks.tar.gz
    """

    def __init__(
        self,
        dataset_id: int,
        seed: Optional[int],
    ):
        self.dataset_name = [
            "slice_localization",
            "protein_structure",
            "naval_propulsion",
            "parkinsons_telemonitoring",
        ][dataset_id]
        data_path = os.path.join(DATA_DIR_NAME, "hpolib", f"fcnet_{self.dataset_name}_data.hdf5")
        self._db = h5py.File(data_path, "r")
        self._rng = np.random.RandomState(seed)
        self._value_range = VALUE_RANGES["hpolib"]

    def __call__(self, config: Dict[str, Union[int, str]], budget: int = 100) -> Dict[str, float]:
        budget = int(budget)
        idx = self._rng.randint(4)
        key = json.dumps({k: self._value_range[k][int(v)] for k, v in config.items()}, sort_keys=True)
        loss = self._db[key]["valid_mse"][idx][budget - 1]
        runtime = self._db[key]["runtime"][idx] * budget / self.max_budget
        return dict(loss=np.log(loss), runtime=runtime)

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._fetch_discrete_config_space()

    @property
    def min_budget(self) -> int:
        return 11

    @property
    def max_budget(self) -> int:
        return 100


class JAHSBench201(AbstractBench):
    _target_metric = "valid-acc"

    def __init__(
        self,
        dataset_id: int,
        seed: Optional[int] = None,  # surrogate is not stochastic
    ):
        # https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar
        # "colorectal_histology" caused memory error, so we do not use it
        self.dataset_name = ["cifar10", "fashion_mnist", "colorectal_histology"][dataset_id]
        data_dir = os.path.join(DATA_DIR_NAME, "jahs_bench_data")
        self._surrogate = jahs_bench.Benchmark(
            task=self.dataset_name, download=False, save_dir=data_dir, metrics=[self._target_metric, "runtime"]
        )
        self._value_range = VALUE_RANGES["jahs-bench"]

    def __call__(self, config: Dict[str, Union[int, str, float]], budget: int = 200) -> Dict[str, float]:
        budget = int(budget)
        EPS = 1e-12
        config = {k: self._value_range[k][int(v)] if k in self._value_range else float(v) for k, v in config.items()}
        assert isinstance(config["LearningRate"], float) and 1e-3 - EPS <= config["LearningRate"] <= 1.0 + EPS
        assert isinstance(config["WeightDecay"], float) and 1e-5 - EPS <= config["WeightDecay"] <= 1e-2 + EPS
        config.update(Optimizer="SGD", Resolution=1.0)
        config = {k: int(v) if k[:-1] == "Op" else v for k, v in config.items()}
        output = self._surrogate(config, nepochs=budget)[budget]
        return dict(loss=100 - output[self._target_metric], runtime=output["runtime"])

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = self._fetch_discrete_config_space()
        config_space.add_hyperparameters(
            [
                CS.UniformFloatHyperparameter(name="LearningRate", lower=1e-3, upper=1.0, log=True),
                CS.UniformFloatHyperparameter(name="WeightDecay", lower=1e-5, upper=1e-2, log=True),
            ]
        )
        return config_space

    @property
    def min_budget(self) -> int:
        return 22

    @property
    def max_budget(self) -> int:
        return 200
