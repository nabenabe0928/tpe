from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np


UNIMODAL = "Unimodal"
MULTIMODAL = "Multimodal"


def config2array(eval_config: Dict[str, float], R: float) -> np.ndarray:
    return np.array(list(eval_config.values())) * R


class AbstractFunc(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sphere(AbstractFunc):
    _R = 5
    name = "Sphere"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        # min: 0
        vals = config2array(eval_config, cls._R)
        return np.sum(vals**2)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return x**2 + y**2


class Styblinski(AbstractFunc):
    _R = 5
    name = "Styblinski"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        # min: -39.166165 * DIM
        vals = config2array(eval_config, R=cls._R)
        t1 = np.sum(vals**4)
        t2 = -16 * np.sum(vals**2)
        t3 = 5 * np.sum(vals)
        return 0.5 * (t1 + t2 + t3)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        t1 = x**4 + y**4
        t2 = -16 * (x**2 + y**2)
        t3 = 5 * (x + y)
        return 0.5 * (t1 + t2 + t3)


class Rastrigin(AbstractFunc):
    _R = 5.12
    name = "Rastrigin"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        # min: 0.0
        dim = len(eval_config)
        vals = config2array(eval_config, R=cls._R)
        t1 = 10 * dim
        t2 = np.sum(vals**2)
        t3 = -10 * np.sum(np.cos(2 * np.pi * vals))
        return t1 + t2 + t3

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        cos_term = np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)
        return 20 + x**2 + y**2 - 10 * cos_term


class Schwefel(AbstractFunc):
    _R = 500
    name = "Schwefel"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        # min: -418.9829 * DIM
        vals = config2array(eval_config, R=cls._R)
        return -np.sum(vals * np.sin(np.sqrt(np.abs(vals))))

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        zx = x * np.sin(np.sqrt(np.abs(x)))
        zy = y * np.sin(np.sqrt(np.abs(y)))
        return -(zx + zy)


class Ackley(AbstractFunc):
    _R = 32.768
    name = "Ackley"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        # min: 0
        vals = config2array(eval_config, R=cls._R)
        t1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(vals**2)))
        t2 = -np.exp(np.mean(np.cos(2 * np.pi * vals)))
        return 20 + np.e + t1 + t2

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        t1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        t2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        return 20 + np.e + t1 + t2


class Griewank(AbstractFunc):
    _R = 600
    name = "Griewank"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        dim = len(eval_config)
        vals = config2array(eval_config, R=cls._R)
        t1 = np.sum(vals**2) / 4000
        t2 = -np.prod(np.cos(vals / np.sqrt(np.arange(1, dim + 1))))
        return 1 + t1 + t2

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        t1 = (x**2 + y**2) / 4000
        t2 = np.cos(x) * np.cos(y / np.sqrt(2))
        return 1 + t1 - t2


class Perm(AbstractFunc):
    _R = 1.0
    _beta = 1.0
    name = "Perm"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        dim = len(eval_config)
        vals = config2array(eval_config, R=cls._R)
        indices = np.arange(dim) + 1
        ret = 0
        for d in indices:
            center = (1 / indices) ** d
            factor = vals**d
            ret += ((indices + cls._beta) @ (factor - center)) ** 2

        return ret

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        z = ((cls._beta + 1) * (x - 1) + (cls._beta + 2) * (y - 0.5)) ** 2 + (
            (cls._beta + 1) * (x**2 - 1) + (cls._beta + 2) * (y - 0.25)
        ) ** 2
        return z


class KTablet(AbstractFunc):
    _R = 5.12
    name = "K-Tablet"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        dim = len(eval_config)
        k = (dim + 3) // 4
        vals = config2array(eval_config, R=cls._R)
        vals[k:] *= 100
        return np.sum(vals**2)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return x**2 + 10000 * y**2


class WeightedSphere(AbstractFunc):
    _R = 5.0
    name = "Weighted sphere"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        dim = len(eval_config)
        vals = config2array(eval_config, R=cls._R)
        weights = np.arange(dim) + 1
        return weights @ (vals**2)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return x**2 + 2 * y**2


class Rosenbrock(AbstractFunc):
    _R = 5.0
    name = "Rosenbrock"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        vals = config2array(eval_config, R=cls._R)
        t1 = np.sum(100 * (vals[1:] - vals[:-1] ** 2) ** 2)
        t2 = np.sum((vals[:-1] - 1) ** 2)
        return t1 + t2

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return 100 * (y - x**2) ** 2 + (x - 1) ** 2


class DifferentPower(AbstractFunc):
    _R = 1.0
    name = "Sum of different power"
    modality = UNIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        dim = len(eval_config)
        vals = config2array(eval_config, R=cls._R)
        indices = np.arange(dim) + 2
        return np.sum(np.abs(vals) ** indices)

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return np.abs(x) ** 2 + np.abs(y) ** 3


class XinSheYang(AbstractFunc):
    _R = 2 * np.pi
    name = "Xin-She Yang"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        vals = config2array(eval_config, R=cls._R)
        return np.sum(np.abs(vals)) * np.exp(-np.sum(np.sin(vals**2)))

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        return (np.abs(x) + np.abs(y)) * np.exp(-(np.sin(x**2) + np.sin(y**2)))


class Levy(AbstractFunc):
    _R = 10.0
    name = "Levy"
    modality = MULTIMODAL

    @classmethod
    def func(cls, eval_config: Dict[str, float]) -> float:
        vals = config2array(eval_config, R=cls._R)
        weights = 1 + (vals - 1) / 4
        t1 = np.sin(np.pi * weights[0]) ** 2
        t2 = (weights[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * weights[-1]) ** 2)
        t3 = np.sum((weights[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * weights[:-1] + 1) ** 2))
        return t1 + t2 + t3

    @classmethod
    def func2d(cls, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = cls._R * X
        y = cls._R * Y
        wx = 1 + (x - 1) / 4
        wy = 1 + (y - 1) / 4
        t1 = np.sin(np.pi * wx) ** 2
        t2 = (wy - 1) ** 2 * (1 + np.sin(2 * np.pi * wy) ** 2)
        t3 = (wx - 1) ** 2 * (1 + 10 * np.sin(np.pi * wx + 1) ** 2)
        return t1 + t2 + t3
