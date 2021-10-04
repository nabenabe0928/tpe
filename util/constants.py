from typing import Union

import ConfigSpace.hyperparameters as CSH

import numpy as np


EPS = 1.0e-12
NumericType = Union[float, int]
SQR2, SQR2PI = np.sqrt(2), np.sqrt(2 * np.pi)

CategoricalHPType = CSH.CategoricalHyperparameter
NumericalHPType = Union[CSH.UniformIntegerHyperparameter, CSH.UniformFloatHyperparameter]

config2type = {
    'UniformFloatHyperparameter': float,
    'UniformIntegerHyperparameter': int
}

type2config = {
    float: 'UniformFloatHyperparameter',
    int: 'UniformIntegerHyperparameter',
    bool: 'CategoricalHyperparameter',
    str: 'CategoricalHyperparameter'
}


def default_percentile(size: int, n_samples_lower: int = 26) -> int:
    return min(int(np.ceil(0.25 * np.sqrt(size))), n_samples_lower)


def default_weights(size: int, n_samples_lower: int = 26) -> np.ndarray:
    weights = np.ones(size)
    return weights / weights.sum()
