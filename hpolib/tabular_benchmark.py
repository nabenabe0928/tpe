import json
import os

from typing import Any, Dict, List, Optional, TypedDict

import h5py
import numpy as np

from hpolib.hyperparameters import BudgetConfig, Hyperparameters


class TabularDataRowType(TypedDict):
    """
    The row data type of the tabular dataset.
    Each row is specified by a string that can be
    casted to dict and this dict is the hyperparameter
    configuration of this row data.

    Attributes:
        final_test_error (List[float]):
            The final test error over 4 seeds
        n_params (List[float]):
            The number of parameters of the model over 4 seeds
        runtime (List[float]):
            The runtime of the model over 4 seeds
        train_loss (List[List[float]]):
            The training loss of the model over 100 epochs
            with 4 different seeds
        train_mse (List[List[float]]):
            The training mse of the model over 100 epochs
            with 4 different seeds
        valid_loss (List[List[float]]):
            The validation loss of the model over 100 epochs
            with 4 different seeds
        valid_mse (List[List[float]]):
            The validation mse of the model over 100 epochs
            with 4 different seeds
    """
    final_test_error: List[float]
    n_params: List[float]
    runtime: List[float]
    train_loss: List[List[float]]
    train_mse: List[List[float]]
    valid_loss: List[List[float]]
    valid_mse: List[List[float]]


DATA_DIR = f'{os.environ["HOME"]}/research/nas_benchmarks/fcnet_tabular_benchmarks/'


class FCNetBenchmark:
    def __init__(
        self,
        path: str = DATA_DIR,
        dataset: str = "fcnet_protein_structure_data.hdf5",
        seed: Optional[int] = None
    ):
        self.data = h5py.File(os.path.join(path, dataset), "r")
        self.rng = np.random.RandomState(seed)

    def get_best_configuration(self) -> Dict[str, Any]:
        """
        Returns:
            The dict of the configuration with the best
            test error and the validation error
            and the test error with this configuration.
        """

        best_test_error, best_key = np.inf, ''
        for key in self.data.keys():
            test_error = np.mean(self.data[key]["final_test_error"])
            if test_error < best_test_error:
                best_test_error = test_error
                best_key = key

        best_config = json.loads(best_key)
        best_val_error = np.mean(self.data[best_key]["valid_mse"][:, -1])
        return {
            "config": best_config,
            "val_error": best_val_error,
            "test_error": best_test_error
        }

    def objective_func(self, config: Dict[str, Any], budget: Dict[str, Any] = {}) -> float:
        """
        Args:
            config (Dict[str, Any]):
                The dict of the configuration and the corresponding value
            budget (Dict[str, Any]):
                The budget information

        Returns:
            val_error (float):
                The validation error given a configuration and a budget.
        """
        _budget = BudgetConfig(**budget)
        config = Hyperparameters(**config).__dict__

        idx = self.rng.randint(4)
        key = json.dumps(config, sort_keys=True)
        return self.data[key]["valid_mse"][idx][_budget.epochs - 1]


class FCNetSliceLocalizationBenchmark(FCNetBenchmark):
    def __init__(self, data_dir: str = DATA_DIR):
        super(FCNetSliceLocalizationBenchmark, self).__init__(
            path=data_dir,
            dataset="fcnet_slice_localization_data.hdf5"
        )


class FCNetProteinStructureBenchmark(FCNetBenchmark):
    def __init__(self, data_dir: str = DATA_DIR):
        super(FCNetProteinStructureBenchmark, self).__init__(
            path=data_dir,
            dataset="fcnet_protein_structure_data.hdf5"
        )


class FCNetNavalPropulsionBenchmark(FCNetBenchmark):
    def __init__(self, data_dir: str = DATA_DIR):
        super(FCNetNavalPropulsionBenchmark, self).__init__(
            path=data_dir,
            dataset="fcnet_naval_propulsion_data.hdf5"
        )


class FCNetParkinsonsTelemonitoringBenchmark(FCNetBenchmark):
    def __init__(self, data_dir: str = DATA_DIR):
        super(FCNetParkinsonsTelemonitoringBenchmark, self).__init__(
            path=data_dir,
            dataset="fcnet_parkinsons_telemonitoring_data.hdf5"
        )
