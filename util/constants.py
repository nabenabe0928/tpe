from typing import Callable, Union

import ConfigSpace.hyperparameters as CSH

import numpy as np


EPS = 1.0e-300
NumericType = Union[float, int]
SQR2, SQR2PI = np.sqrt(2), np.sqrt(2 * np.pi)

CategoricalHPType = CSH.CategoricalHyperparameter
NumericalHPType = Union[
    CSH.UniformIntegerHyperparameter,
    CSH.UniformFloatHyperparameter,
    CSH.OrdinalHyperparameter,
]

config2type = {
    "UniformFloatHyperparameter": float,
    "UniformIntegerHyperparameter": int,
    "OrdinalHyperparameter": float,
}

type2config = {
    float: "UniformFloatHyperparameter",
    int: "UniformIntegerHyperparameter",
    bool: "CategoricalHyperparameter",
    str: "CategoricalHyperparameter",
}


def default_percentile_maker() -> Callable[[np.ndarray], int]:
    def _imp(vals: np.ndarray) -> int:
        size = vals.size
        return int(np.ceil(0.25 * np.sqrt(size)))

    return _imp


def uniform_weight(size: int) -> np.ndarray:
    return np.full(size, 1.0 / size)
