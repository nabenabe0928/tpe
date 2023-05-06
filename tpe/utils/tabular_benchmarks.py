import json
import os
import pickle
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, TypedDict, Union

import ConfigSpace as CS

try:
    import jahs_bench
except ModuleNotFoundError:  # We cannot use jahs with smac
    pass

import numpy as np

from yahpo_gym import benchmark_set, local_config


DATA_DIR_NAME = os.path.join(os.environ["HOME"], "tabular_benchmarks")
VALUE_RANGES = json.load(open("tpe/utils/tabular_benchmarks.json"))


class RowDataType(TypedDict):
    valid_mse: List[Dict[int, float]]
    runtime: List[float]


class BaseBenchData:
    NotImplemented


class HPOLibDatabase(BaseBenchData):
    """ Workaround to prevent dask from serializing the objective func """
    def __init__(self, dataset_name: str):
        data_path = os.path.join(DATA_DIR_NAME, "hpolib", f"{dataset_name}.pkl")
        self._db = pickle.load(open(data_path, "rb"))

    def __getitem__(self, key: str) -> Dict[str, RowDataType]:
        return self._db[key]


class LCBenchSurrogate(BaseBenchData):
    """ Workaround to prevent dask from serializing the objective func """
    def __init__(self, dataset_id: str, target_metric: str):
        self._target_metric = target_metric
        self._dataset_id = dataset_id
        self._surrogate = benchmark_set.BenchmarkSet("lcbench", instance=dataset_id, active_session=False)

    def __call__(self, config: Dict[str, Union[int, float]], budget: int) -> Dict[str, float]:
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()

        config["OpenML_task_id"] = self._dataset_id
        config["epoch"] = budget
        output = self._surrogate.objective_function(config)[0]
        return dict(loss=1.0 - output[self._target_metric], runtime=output["time"])


class JAHSBenchSurrogate(BaseBenchData):
    """ Workaround to prevent dask from serializing the objective func """
    def __init__(self, data_dir: str, dataset_name: str, target_metric):
        self._target_metric = target_metric
        self._surrogate = jahs_bench.Benchmark(
            task=dataset_name, download=False, save_dir=data_dir, metrics=[self._target_metric, "runtime"]
        )

    def __call__(self, config: Dict[str, Union[int, str, float]], budget: int = 200) -> Dict[str, float]:
        config.update(Optimizer="SGD", Resolution=1.0)
        config = {k: int(v) if k[:-1] == "Op" else v for k, v in config.items()}
        output = self._surrogate(config, nepochs=budget)[budget]
        return dict(loss=100 - output[self._target_metric], runtime=output["runtime"])


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

    @abstractmethod
    def get_data(self) -> BaseBenchData:
        raise NotImplementedError

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
    _N_DATASETS = 34
    _DATASET_NAMES = (
        "kddcup09",
        "covertype",
        "amazon-employee-access",
        "adult",
        "nomao",
        "bank-marketing",
        "shuttle",
        "australian",
        "kr-vs-kp",
        "mfeat-factors",
        "credit-g",
        "vehicle",
        "kc1",
        "blood-transfusion-service-center",
        "cnae-9",
        "phoneme",
        "higgs",
        "connect-4",
        "helena",
        "jannis",
        "volkert",
        "mini-boo-ne",
        "aps-failure",
        "christine",
        "fabert",
        "airlines",
        "jasmine",
        "sylvine",
        "albert",
        "dionis",
        "car",
        "segment",
        "fashion-mnist",
        "jungle-chess-2pcs-raw-endgame-complete",
    )

    def __init__(
        self,
        dataset_id: int,
        seed: Optional[int] = None,  # surrogate is not stochastic
        keep_benchdata: bool = True,
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
        self._surrogate = self.get_data() if keep_benchdata else None
        self._config_space = self.config_space

    def get_data(self) -> LCBenchSurrogate:
        return LCBenchSurrogate(dataset_id=self._dataset_id, target_metric=self._target_metric)

    def _validate_config(self, config: Dict[str, Union[int, float]]) -> None:
        EPS = 1e-12
        for hp in self._config_space.get_hyperparameters():
            lb, ub, name = hp.lower, hp.upper, hp.name
            if isinstance(hp, CS.UniformFloatHyperparameter):
                assert isinstance(config[name], float) and lb - EPS <= config[name] <= ub + EPS
            else:
                config[name] = int(config[name])
                assert isinstance(config[name], int) and lb <= config[name] <= ub

    def __call__(
        self,
        config: Dict[str, Union[int, float]],
        budget: int = 52,
        seed: Optional[int] = None,
        bench_data: Optional[LCBenchSurrogate] = None
    ) -> Dict[str, float]:
        if bench_data is None and self._surrogate is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        surrogate = bench_data if self._surrogate is None else self._surrogate
        budget = int(min(self._TRUE_MAX_BUDGET, budget))
        self._validate_config(config)
        return surrogate(config, budget)

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

    Use https://github.com/nabenabe0928/hpolib-extractor to extract the pickle file.
    """
    _N_DATASETS = 4
    _DATASET_NAMES = ("slice-localization", "protein-structure",  "naval-propulsion", "parkinsons-telemonitoring")

    def __init__(
        self,
        dataset_id: int,
        seed: Optional[int],
        keep_benchdata: bool = True,
    ):
        self.dataset_name = [
            "slice_localization",
            "protein_structure",
            "naval_propulsion",
            "parkinsons_telemonitoring",
        ][dataset_id]
        self._db = self.get_data() if keep_benchdata else None
        self._rng = np.random.RandomState(seed)
        self._value_range = VALUE_RANGES["hpolib"]

    def get_data(self) -> HPOLibDatabase:
        return HPOLibDatabase(self.dataset_name)

    def __call__(
        self,
        config: Dict[str, Union[int, str]],
        budget: int = 100,
        seed: Optional[int] = None,
        bench_data: Optional[HPOLibDatabase] = None
    ) -> Dict[str, float]:
        if bench_data is None and self._db is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        db = bench_data if self._db is None else self._db
        budget = int(budget)
        idx = seed % 4 if seed is not None else self._rng.randint(4)
        key = json.dumps({k: self._value_range[k][int(v)] for k, v in config.items()}, sort_keys=True)
        loss = db[key]["valid_mse"][idx][budget - 1]
        runtime = db[key]["runtime"][idx] * budget / self.max_budget
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
    _N_DATASETS = 3
    _DATASET_NAMES = ("cifar10", "fashion-mnist", "colorectal-histology")

    def __init__(
        self,
        dataset_id: int,
        seed: Optional[int] = None,  # surrogate is not stochastic
        keep_benchdata: bool = True,
    ):
        # https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar
        # "colorectal_histology" caused memory error, so we do not use it
        self.dataset_name = ["cifar10", "fashion_mnist", "colorectal_histology"][dataset_id]
        self._data_dir = os.path.join(DATA_DIR_NAME, "jahs_bench_data")
        self._surrogate = self.get_data() if keep_benchdata else None
        self._value_range = VALUE_RANGES["jahs-bench"]

    def get_data(self) -> JAHSBenchSurrogate:
        return JAHSBenchSurrogate(
            data_dir=self._data_dir, dataset_name=self.dataset_name, target_metric=self._target_metric
        )

    def __call__(
        self,
        config: Dict[str, Union[int, str, float]],
        budget: int = 200,
        seed: Optional[int] = None,
        bench_data: Optional[JAHSBenchSurrogate] = None
    ) -> Dict[str, float]:
        if bench_data is None and self._surrogate is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        surrogate = bench_data if self._surrogate is None else self._surrogate
        budget = int(budget)
        EPS = 1e-12
        config = {k: self._value_range[k][int(v)] if k in self._value_range else float(v) for k, v in config.items()}
        assert isinstance(config["LearningRate"], float) and 1e-3 - EPS <= config["LearningRate"] <= 1.0 + EPS
        assert isinstance(config["WeightDecay"], float) and 1e-5 - EPS <= config["WeightDecay"] <= 1e-2 + EPS
        return surrogate(config, budget)

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


if __name__ == "__main__":
    local_config.init_config()
    local_config.set_data_path(DATA_DIR_NAME)
