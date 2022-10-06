from typing import List, Union

import ConfigSpace.hyperparameters as CSH

import numpy as np

from scipy.stats import rankdata


EPS = 1.0e-300
NumericType = Union[float, int]
SQR2, SQR2PI = np.sqrt(2), np.sqrt(2 * np.pi)

CategoricalHPType = Union[CSH.CategoricalHyperparameter]
NumericalHPType = Union[
    CSH.UniformIntegerHyperparameter,
    CSH.UniformFloatHyperparameter,
    CSH.OrdinalHyperparameter,
]
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


def default_quantile(size: int) -> int:
    return int(np.ceil(0.25 * np.sqrt(size)))


def default_weight(size: int) -> np.ndarray:
    return np.full(size, 1.0 / size)


class WeightFuncs:
    choices: List[str] = ["uniform", "older-smaller", "weaker-smaller", "expected-improvement"]
    _decay_start_from = 25

    @classmethod
    def _decayed_weight(self, size: int) -> np.ndarray:
        flat = np.ones(self._decay_start_from)
        n_decays = size - self._decay_start_from
        ramp = np.linspace(1.0 / size, 1.0, num=n_decays)
        weights = np.concatenate([ramp, flat], axis=0)
        return weights / np.sum(weights)

    @classmethod
    def uniform(self, size: int) -> np.ndarray:
        return np.full(size, 1.0 / size)

    @classmethod
    def older_smaller(self, order: np.ndarray) -> np.ndarray:
        if order.size < self._decay_start_from:
            return self.uniform(order.size)

        ages = rankdata(order, method="ordinal") - 1
        weights = self._decayed_weight(order.size)
        return weights[ages]

    @classmethod
    def weaker_smaller(self, sorted_loss_vals: np.ndarray) -> np.ndarray:
        if sorted_loss_vals.size < self._decay_start_from:
            return self.uniform(sorted_loss_vals.size)

        weights = self._decayed_weight(sorted_loss_vals.size)
        return weights[::-1]

    @classmethod
    def expected_improvement(self, sorted_loss_vals: np.ndarray) -> np.ndarray:
        threshold = sorted_loss_vals.max()
        weights = threshold - sorted_loss_vals
        weights /= np.sum(weights)
        return weights


class QuantileFunc:
    choices: List[str] = ["linear", "sqrt"]

    def __init__(self, alpha: float = 0.25, choice: str = "sqrt"):
        self._alpha = alpha
        self._choice = choice

    def __call__(self, size: int) -> int:
        if self._choice == "linear":
            return self.linear(size)
        elif self._choice == "sqrt":
            return self.sqrt(size)
        else:
            raise ValueError(f"choice must be in {self.choices}, but got {self._choice}")

    def sqrt(self, size: int) -> int:
        return int(np.ceil(self._alpha * np.sqrt(size)))

    def linear(self, size: int) -> int:
        return int(np.ceil(self._alpha * size))
