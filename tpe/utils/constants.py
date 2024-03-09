from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, Union

import ConfigSpace.hyperparameters as CSH

import numpy as np


EPS = 1.0e-300
NumericType = Union[float, int]
SQR2, SQR2PI = np.sqrt(2), np.sqrt(2 * np.pi)

CategoricalHPType = CSH.CategoricalHyperparameter
NumericalHPType = Union[CSH.UniformIntegerHyperparameter, CSH.UniformFloatHyperparameter, CSH.OrdinalHyperparameter]
HPType = Union[CategoricalHPType, NumericalHPType]

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


class PercentileFuncMaker(Protocol):
    def __call__(self, **kwargs: dict[str, Any]) -> Callable[[np.ndarray], int]:
        raise NotImplementedError


def default_percentile_maker() -> Callable[[np.ndarray], int]:
    def _imp(vals: np.ndarray) -> int:
        size = vals.size
        return int(np.ceil(0.25 * np.sqrt(size)))

    return _imp
