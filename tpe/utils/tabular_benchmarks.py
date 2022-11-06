import json
import os
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Final, List, Optional, Union

import ConfigSpace as CS

import h5py

import jahs_bench

import numpy as np

import pyarrow.parquet as pq  # type: ignore


DATA_DIR_NAME = os.path.join(os.environ["HOME"], "tabular_benchmarks")
SEEDS: Final = [665, 1319, 7222, 7541, 8916]
VALUE_RANGES = json.load(open("tpe/utils/tabular_benchmarks.json"))


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
                if not isinstance(choices[0], str)
                else CS.CategoricalHyperparameter(name=name, choices=[str(i) for i in range(len(choices))])
                for name, choices in self._value_range.items()
            ]
        )
        return config_space

    @property
    @abstractmethod
    def config_space(self) -> CS.ConfigurationSpace:
        raise NotImplementedError


class HPOBench(AbstractBench):
    def __init__(
        self,
        dataset_id: int,
        seed: Optional[int],
    ):
        dataset_info = [
            ("credit_g", 31),
            ("vehicle", 53),
            ("kc1", 3917),
            ("phoneme", 9952),
            ("blood_transfusion", 10101),
            ("australian", 146818),
            ("car", 146821),
            ("segment", 146822),
        ]
        self.dataset_name, dataset_id = dataset_info[dataset_id]
        budget_name = "iter"
        data_path = os.path.join(DATA_DIR_NAME, "hpo-bench", str(dataset_id), f"nn_{dataset_id}_data.parquet.gzip")
        db = pq.read_table(data_path, filters=[(budget_name, "==", 243)])
        self._db = db.drop([budget_name, "subsample"])
        self._rng = np.random.RandomState(seed)
        self._value_range = VALUE_RANGES["hpo-bench"]

    def _validate_query(self, query: Dict[str, Any], config: Dict[str, Union[int, float]]) -> None:
        if len(query["__index_level_0__"]) != 1:
            raise ValueError(f"There must be only one row for config={config}, but got query={query}")

        queried_config = {k: query[k][0] for k in config.keys()}
        if not all(np.isclose(queried_config[k], v, rtol=1e-3) for k, v in config.items()):
            raise ValueError(f"The query must have the identical config as {config}, but got {queried_config}")

    def __call__(self, config: Dict[str, int]) -> float:
        config["seed"] = SEEDS[self._rng.randint(len(SEEDS))]
        KEY_ORDER = ["alpha", "batch_size", "depth", "learning_rate_init", "width", "seed"]

        assert len(config) == len(KEY_ORDER)
        idx = 0
        eval_config: Dict[str, Union[int, float]] = {}
        for k in KEY_ORDER:
            true_val = self._value_range[k][config[k]] if k != "seed" else config[k]
            assert not isinstance(true_val, str)  # mypy redefinition
            eval_config[k] = true_val
            idx = self._db[k].index(true_val, start=idx).as_py()

        query = self._db.take([idx]).to_pydict()
        self._validate_query(query, eval_config)
        return 1.0 - query["result"][0]["info"]["val_scores"]["bal_acc"]

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._fetch_discrete_config_space()


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

    def __call__(self, config: Dict[str, Union[int, str]]) -> float:
        idx = self._rng.randint(4)
        key = json.dumps({k: self._value_range[k][int(v)] for k, v in config.items()}, sort_keys=True)
        return np.log(self._db[key]["valid_mse"][idx][99])

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._fetch_discrete_config_space()


class JAHSBench201(AbstractBench):
    def __init__(
        self,
        dataset_id: int,
        seed: Optional[int] = None,  # surrogate is not stochastic
    ):
        # https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.0.0/assembled_surrogates.tar
        # "colorectal_histology" caused memory error, so we do not use it
        self.dataset_name = ["cifar10", "fashion_mnist"][dataset_id]
        data_dir = os.path.join(DATA_DIR_NAME, "jahs_bench_data")
        self._surrogate = jahs_bench.Benchmark(task=self.dataset_name, download=False, save_dir=data_dir)
        self._value_range = VALUE_RANGES["jahs-bench"]

    def __call__(self, config: Dict[str, Union[int, str, float]]) -> float:
        EPS = 1e-12
        config = {k: self._value_range[k][int(v)] if k in self._value_range else float(v) for k, v in config.items()}
        assert isinstance(config["LearningRate"], float) and 1e-3 - EPS <= config["LearningRate"] <= 1.0 + EPS
        assert isinstance(config["WeightDecay"], float) and 1e-5 - EPS <= config["WeightDecay"] <= 1e-2 + EPS
        config.update(Optimizer="SGD", Resolution=1.0)
        config = {k: int(v) if k[:-1] == "Op" else v for k, v in config.items()}
        return 100 - self._surrogate(config, nepochs=200)[200]["valid-acc"]

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
