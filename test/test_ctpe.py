import pytest
import time
import unittest
from typing import Dict, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.optimizer import TPEOptimizer


def small2d(eval_config: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Simulation 2 from the paper:
        "Bayesian Optimization with Inequality Constraints"
        http://proceedings.mlr.press/v32/gardner14.pdf
    """
    start = time.time()
    loss = np.sin(eval_config["x0"]) + eval_config["x1"]
    constraint = np.sin(eval_config["x0"]) * np.sin(eval_config["x1"])
    return {"loss": loss, "c": constraint}, time.time() - start


def dummy_func(eval_config: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    rnd = np.random.random(3)
    return {"loss": rnd[0], "c1": rnd[1], "c2": rnd[2]}, 0.0


def dummy_mo_func(eval_config: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    rnd = np.random.random(4)
    return {"loss": rnd[0], "f2": rnd[1], "c1": rnd[2], "c2": rnd[3]}, 0.0


def _get_default_opt() -> TPEOptimizer:
    dim = 2
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=0, upper=6))

    opt = TPEOptimizer(
        obj_func=small2d,
        config_space=cs,
        min_bandwidth_factor=1e-2,
        max_evals=50,
        constraints={"c": -0.95},
    )
    return opt


def _get_default_opt_with_multi_constraints() -> TPEOptimizer:
    dim = 2
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=0, upper=6))

    opt = TPEOptimizer(
        obj_func=dummy_func,
        config_space=cs,
        min_bandwidth_factor=1e-2,
        max_evals=50,
        constraints={"c1": 0.5, "c2": 0.5},
    )
    return opt


def _get_default_multi_opt_with_multi_constraints() -> TPEOptimizer:
    dim = 2
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=0, upper=6))

    opt = TPEOptimizer(
        obj_func=dummy_mo_func,
        config_space=cs,
        min_bandwidth_factor=1e-2,
        max_evals=50,
        objective_names=["loss", "f2"],
        constraints={"c1": 0.5, "c2": 0.5},
    )
    return opt


def test_optimize() -> None:
    opt = _get_default_opt()
    opt.optimize()
    data = opt.fetch_observations()
    assert np.sum(data["c"] <= -0.95) == opt._sampler._satisfied_flag.sum()
    assert data["c"].size == opt._sampler._satisfied_flag.size
    assert opt._sampler._feasible_counts["c"] == np.sum(data["c"] <= -0.95)

    opt = _get_default_opt_with_multi_constraints()
    opt.optimize()
    data = opt.fetch_observations()
    assert np.sum((data["c1"] <= 0.5) & (data["c2"] <= 0.5)) == opt._sampler._satisfied_flag.sum()
    assert data["c1"].size == opt._sampler._satisfied_flag.size
    assert opt._sampler._feasible_counts["c1"] == np.sum(data["c1"] <= 0.5)
    assert opt._sampler._feasible_counts["c2"] == np.sum(data["c2"] <= 0.5)


def test_optimize_with_knowledge_augmentation() -> None:
    opt = _get_default_opt_with_multi_constraints()
    size = 50
    observations = {key: np.random.random(size) for key in ["x0", "x1", "c1"]}
    opt.apply_knowledge_augmentation(observations)
    opt.optimize()

    data = opt.fetch_observations()
    assert np.sum((data["c1"] <= 0.5) & (data["c2"] <= 0.5)) == opt._sampler._satisfied_flag.sum()
    assert data["c1"].size == opt._sampler._satisfied_flag.size
    assert opt._sampler._feasible_counts["c1"] == np.sum(data["c1"] <= 0.5) + np.sum(observations["c1"] <= 0.5)
    assert opt._sampler._feasible_counts["c2"] == np.sum(data["c2"] <= 0.5)


def test_is_satisfied() -> None:
    opt = _get_default_opt_with_multi_constraints()
    assert opt._sampler._is_satisfied({"c1": 0.4, "c2": 0.4})
    assert not opt._sampler._is_satisfied({"c1": 0.4, "c2": 0.6})
    assert not opt._sampler._is_satisfied({"c1": 0.6, "c2": 0.4})
    assert not opt._sampler._is_satisfied({"c1": 0.6, "c2": 0.6})


def test_observations() -> None:
    opt = _get_default_multi_opt_with_multi_constraints()
    max_evals = opt._max_evals
    opt.optimize()
    data = opt.fetch_observations()
    for name in opt._hp_names:
        assert data[name].size == max_evals
    for name in opt._metric_names:
        assert data[name].size == max_evals

    size = 50
    observations = {key: np.random.random(size) for key in ["x0", "x1", "c2"]}
    opt = _get_default_opt_with_multi_constraints()
    opt.apply_knowledge_augmentation(observations)
    opt.optimize()
    data = opt.fetch_observations()
    for name in opt._hp_names:
        assert data[name].size == max_evals
    for name in opt._metric_names:
        assert data[name].size == max_evals


def test_get_config_candidates() -> None:
    opt = _get_default_opt_with_multi_constraints()
    opt.optimize()
    n_ei_samples = opt._sampler._n_ei_candidates
    assert opt._sampler.get_config_candidates()["x0"].size == n_ei_samples * 3
    opt = _get_default_multi_opt_with_multi_constraints()
    opt.optimize()
    n_ei_samples = opt._sampler._n_ei_candidates
    assert opt._sampler.get_config_candidates()["x0"].size == n_ei_samples * 3


def test_apply_knowledge_augmentation() -> None:
    size = 50
    observations = {key: np.random.random(size) for key in ["x0", "x1", "loss", "c1", "c2"]}
    for key in ["c2", "loss", None]:
        opt = _get_default_opt_with_multi_constraints()
        opt.apply_knowledge_augmentation(observations)
        all_constraints_exist = all(key in observations for key in ["c1", "c2"])
        assert opt._sampler._satisfied_flag.size == (all_constraints_exist * 50)
        for name in ["c1", "c2"]:
            if name in observations:
                assert opt._sampler._feasible_counts[name] > 0
            else:
                assert opt._sampler._feasible_counts[name] == 0

        if key is not None:
            observations.pop(key)

    opt = _get_default_multi_opt_with_multi_constraints()
    observations = {key: np.random.random(size) for key in ["x0", "x1", "loss", "f2", "c1", "c2"]}
    opt.apply_knowledge_augmentation(observations)

    with pytest.raises(ValueError):
        opt.apply_knowledge_augmentation(observations)

    opt = _get_default_multi_opt_with_multi_constraints()
    observations = {key: np.random.random(size) for key in ["x0", "x1", "loss", "f2", "c1", "c2"]}
    with pytest.raises(ValueError):
        observations["x0"] = observations["x0"][:-4]  # reduce one of the data columns
        opt.apply_knowledge_augmentation(observations)


def test_get_is_satisfied_flag_from_data() -> None:
    opt = _get_default_opt_with_multi_constraints()
    flag = opt._sampler._get_is_satisfied_flag_from_data(
        n_observations=4,
        observations={"c1": np.array([0.4, 0.4, 0.6, 0.6]), "c2": np.array([0.4, 0.6, 0.4, 0.6])},
    )
    assert np.all(flag == np.array([True, False, False, False]))


def test_init_samplers() -> None:
    opt = _get_default_multi_opt_with_multi_constraints()
    assert all(key in opt._sampler._samplers for key in ["objective", "c1", "c2"])


def test_percentile_func_for_objective() -> None:
    pass


if __name__ == "__main__":
    unittest.main()