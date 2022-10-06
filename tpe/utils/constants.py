from typing import List, Literal, Protocol, Union

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


class WeightFuncType(Protocol):
    def __call__(self, size: int, order: np.ndarray, sorted_loss_vals: np.ndarray, lower_group: bool) -> np.ndarray:
        raise NotImplementedError


class WeightFuncs:
    choices: List[str] = ["uniform", "older-smaller", "weaker-smaller", "expected-improvement"]
    _decay_start_from = 25

    def __init__(self, choice: Literal["uniform", "older-smaller", "weaker-smaller", "expected-improvement"]):
        self._choice = choice
        if choice not in self.choices:
            raise ValueError(f"choice must be in {self.choices}, but got {self._choice}")

    def __call__(
        self,
        size: int,
        order: np.ndarray,
        sorted_loss_vals: np.ndarray,
        lower_group: bool = False,
    ) -> np.ndarray:
        # NOTE: We always add a weight for prior and thus weights.size is always size + 1
        if self._choice == "uniform":
            return self.uniform(size)
        elif self._choice == "older-smaller":
            return self.older_smaller(order)
        elif self._choice == "weaker-smaller":
            return self.weaker_smaller(sorted_loss_vals)
        elif self._choice == "expected-improvement":
            return self.expected_improvement(sorted_loss_vals) if lower_group else self.uniform(size)
        else:
            raise NotImplementedError(f"Unknown choice {self._choice}")

    @classmethod
    def _decayed_weight(cls, size: int) -> np.ndarray:
        flat = np.ones(cls._decay_start_from)
        n_decays = size - cls._decay_start_from
        ramp = np.linspace(1.0 / size, 1.0, num=n_decays)
        weights = np.concatenate([ramp, flat], axis=0)
        return weights / np.sum(weights)

    @classmethod
    def uniform(cls, size: int) -> np.ndarray:
        size += 1  # for prior
        return np.full(size, 1.0 / size)

    @classmethod
    def older_smaller(cls, order: np.ndarray) -> np.ndarray:
        if order.size < cls._decay_start_from:
            return cls.uniform(order.size)

        order = np.append(order, -1)  # prior is the oldest
        ages = rankdata(order, method="ordinal") - 1
        weights = cls._decayed_weight(order.size)
        return weights[ages]

    @classmethod
    def weaker_smaller(cls, sorted_loss_vals: np.ndarray) -> np.ndarray:
        if sorted_loss_vals.size < cls._decay_start_from:
            return cls.uniform(sorted_loss_vals.size)

        weights = cls._decayed_weight(sorted_loss_vals.size + 1)
        return weights[::-1]  # prior is the weakest

    @classmethod
    def expected_improvement(cls, sorted_loss_vals: np.ndarray) -> np.ndarray:
        threshold = np.max(sorted_loss_vals)
        weights = threshold - sorted_loss_vals
        # prior is 1.0 / size (uniform weight)
        weights = np.append(weights, np.mean(weights))
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
