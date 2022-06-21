import pytest
import time
import unittest
from typing import Dict, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.optimizer import TPEOptimizer


def multi_modal_MOP(config: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    multi-modal MOP

    Optimal at:
        f1 = x1
        f2 = 1 - sqrt(f1)
    """
    DIM = 2
    start = time.time()
    X = np.array([config[f"x{d}"] for d in range(DIM)])
    f1 = X[0]
    g = 1 + 10 * (DIM - 1) + np.sum(X[1:] ** 2 - 10 * np.cos(2 * np.pi * X[1:]))
    h = 1 - np.sqrt(f1 / g) if f1 <= g else 0
    return {"f1": f1, "f2": g * h}, time.time() - start


def _get_default_opt() -> TPEOptimizer:
    DIM = 2
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("x0", 0, 1))
    V = 0.5  # 30
    for d in range(1, DIM):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-V, upper=V))

    max_evals = 50
    opt = TPEOptimizer(
        obj_func=multi_modal_MOP,
        config_space=cs,
        objective_names=["f1", "f2"],
        min_bandwidth_factor=1e-3,
        n_init=10,
        max_evals=max_evals,
    )
    return opt


def test_by_multi_modal_MOP() -> None:
    opt = _get_default_opt()
    max_evals = opt._max_evals
    opt.optimize()
    assert opt.fetch_observations()["f1"].size == max_evals
    assert opt.fetch_observations()["f2"].size == max_evals
    assert opt._sampler._percentile_func() == max(opt._sampler._n_fronts, (15 * 50 + 99) // 100)


def test_calculate_order() -> None:
    opt = _get_default_opt()
    opt._sampler._observations["f1"] = np.array([0, 2, 3, 5])
    opt._sampler._observations["f2"] = np.array([3, 1, 2, 4])
    order = opt._sampler._calculate_order()
    assert np.all(order == [1, 0, 2, 3])

    with pytest.raises(ValueError):
        opt._sampler._calculate_order({})


def test_apply_knowledge_augmentation() -> None:
    opt = _get_default_opt()
    observations = {name: np.random.random(40) * 0.5 for name in ["x0", "x1", "f1", "f2"]}
    opt.apply_knowledge_augmentation(observations)
    with pytest.raises(ValueError):
        opt = _get_default_opt()
        observations.pop("f1")
        opt.apply_knowledge_augmentation(observations)


if __name__ == "__main__":
    unittest.main()
