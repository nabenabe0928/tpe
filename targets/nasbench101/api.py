import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import numpy as np

from nasbench import api

from targets.base_tabularbench_api import BaseTabularBenchAPI


MAX_EDGES = 9
MAX_VERTICES = 7
N_ARCH = 423624
AVAIL_SEEDS = [0, 1, 2]
DATA_DIR = f'{os.environ["HOME"]}/tabular_benchmarks/nasbench101/'


class StaticDataRowType(TypedDict):
    """
    The row data type of the static information given a configuration.

    Attributes:
        module_adjacency (np.ndarray):
            The adjacency matrix that represents the connections
            between operations.
            Note that since the graph assumes directed acyclic graph (DAG),
            The edge direction is always the i-th operation to the j-th operation
            where i < j.
        module_operations (List[Literal['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']]):
            The list of operations.
            The first element and the last element must be `input` and `output`, respectively.
        trainable_parameters (int):
            The number of parameters that can be trained.
    """

    module_adjacency: np.ndarray
    module_operations: List[Literal["input", "conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3", "output"]]
    trainable_parameters: int


class NonStaticDataRowType(TypedDict):
    """
    The row data type of the non-static information given a configuration.
    This row data type is mapped by
    Dict[epochs (int in {108} or {4, 12, 36, 108})][seed (int in [0, 2])]
        ==> row data type.
    Tabular data will have epochs of 108 only when we specify `only108` dataset.

    Attributes:
        halfway_training_time (float):
            The runtime for the training with epochs / 2.
        halfway_train_accuracy (float):
            The training accuracy for the training with epochs / 2.
            The value range is [0, 1].
        halfway_validation_accuracy (float):
            The validation accuracy for the training with epochs / 2.
            The value range is [0, 1].
        halfway_test_accuracy (float):
            The test accuracy for the training with epochs / 2.
            The value range is [0, 1].
        final_training_time (float):
            The runtime for the training with epochs.
        final_train_accuracy (float):
            The training accuracy for the training with epochs.
            The value range is [0, 1].
        final_validation_accuracy (float):
            The validation accuracy for the training with epochs.
            The value range is [0, 1].
        final_test_accuracy (float):
            The test accuracy for the training with epochs.
            The value range is [0, 1].
    """

    halfway_training_time: float
    halfway_train_accuracy: float
    halfway_validation_accuracy: float
    halfway_test_accuracy: float
    final_training_time: float
    final_train_accuracy: float
    final_validation_accuracy: float
    final_test_accuracy: float


class DatasetChoices(Enum):
    full = "nasbench_full.tfrecord"
    only108 = "nasbench_only108.tfrecord"


class SearchSpaceChoices(Enum):
    # https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py
    cifar10A = "cifar10A"
    cifar10B = "cifar10B"
    cifar10C = "cifar10C"


class NASBench101(BaseTabularBenchAPI):
    """
    Download the datasets via:
        # Full dataset (2GB)
        $ wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
        # Only108 dataset (0.5GB)
        $ wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
    """

    def __init__(
        self,
        path: str = DATA_DIR,
        dataset: Union[DatasetChoices, str] = DatasetChoices.only108,
        seed: Optional[int] = None,
        search_space: Union[SearchSpaceChoices, str] = SearchSpaceChoices.cifar10C,
    ):

        dataset = self._validate_choice(dataset, DatasetChoices)
        search_space = self._validate_choice(search_space, SearchSpaceChoices)
        super().__init__(
            hp_module_path=f"targets/nasbench101/{search_space.value}", dataset_name=search_space.name, seed=seed
        )

        module_path = f"targets.nasbench101.{search_space.value}"
        hp_path = f"{module_path}.hyperparameters"
        self._Hyperparameters = getattr(__import__(hp_path, fromlist=["", ""]), "Hyperparameters")
        self._BudgetConfig = getattr(__import__(hp_path, fromlist=["", ""]), "BudgetConfig")
        self._config2spec = getattr(__import__(hp_path, fromlist=["", ""]), "config2spec")

        self._data = api.NASBench(os.path.join(path, dataset.value))

    def fetch_all_losses(self) -> np.ndarray:
        """the oracle results are available in targets/nasbench101/cifar10A/constraints.json"""
        cnt = 0
        loss_vals = np.empty(N_ARCH * len(AVAIL_SEEDS))

        for hash in self.data.hash_iterator():
            static_info, non_static_info = self.data.get_metrics_from_hash(hash)
            for i, info in enumerate(non_static_info[108]):  # TODO: Change budget
                loss_vals[cnt] = 1 - info["final_validation_accuracy"]
                cnt += 1
        return loss_vals

    def objective_func(self, config: Dict[str, Any], budget: Dict[str, Any] = {}) -> float:
        idx = self._rng.randint(3)
        _config = self._Hyperparameters(**config)
        _budget = self._BudgetConfig(**budget)

        model_spec = self._config2spec(_config)

        # get_metrics_from_spec(api.ModelSpec: model_spec) -> Tuple[NonStaticDataRowType, StaticDataRowType]
        try:
            _, non_static_info = self.data.get_metrics_from_spec(model_spec)
        except api.OutOfDomainError:
            return 1.0

        return 1.0 - non_static_info[_budget.epochs][idx]["final_validation_accuracy"]

    @property
    def data(self) -> api.NASBench:
        return self._data
