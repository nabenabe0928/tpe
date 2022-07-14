import pytest
import time
import unittest
from typing import Dict, Optional, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.optimizer import TPEOptimizer
from tpe.optimizer.models import MetaLearnTPE
from tpe.utils.constants import OBJECTIVE_KEY


def sphere(eval_config: Dict[str, float], shift: float = 0.0) -> Tuple[Dict[str, float], float]:
    vals = np.array(list(eval_config.values())) - shift
    vals *= vals
    return {"loss": vals.sum()}, 0.0


def multi_modal_MOP(eval_config: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    multi-modal MOP

    Optimal at:
        f1 = x1
        f2 = 1 - sqrt(f1)
    """
    DIM = 2
    start = time.time()
    X = np.array([eval_config[f"x{d}"] for d in range(DIM)])
    f1 = X[0]
    g = 1 + 10 * (DIM - 1) + np.sum(X[1:] ** 2 - 10 * np.cos(2 * np.pi * X[1:]))
    h = 1 - np.sqrt(f1 / g) if f1 <= g else 0
    return {"f1": f1, "f2": g * h}, time.time() - start


def get_metadata(dim: int) -> Dict[str, Dict[str, np.ndarray]]:
    n_tasks, n_evals = 3, 50
    metadata: Dict[str, Dict[str, np.ndarray]] = {
        f"shift={i}": {f"x{d}": np.random.random(n_evals) for d in range(dim)} for i in range(1, n_tasks)
    }
    for shift, task_name in enumerate(metadata.keys(), start=1):
        metadata[task_name]["loss"] = np.array(
            [
                sphere({f"x{d}": metadata[task_name][f"x{d}"][i] for d in range(dim)}, shift=shift)[0]["loss"]
                for i in range(n_evals)
            ]
        )

    return metadata


def get_mo_metadata(dim: int) -> Dict[str, Dict[str, np.ndarray]]:
    n_tasks, n_evals = 2, 50
    metadata: Dict[str, Dict[str, np.ndarray]] = {
        "source": {f"x{d}": np.random.random(n_evals) for d in range(dim)} for i in range(1, n_tasks)
    }
    for task_name in metadata.keys():
        results = [
            multi_modal_MOP({f"x{d}": metadata[task_name][f"x{d}"][i] for d in range(dim)})[0] for i in range(n_evals)
        ]
        metadata[task_name]["f1"] = np.array([res["f1"] for res in results])
        metadata[task_name]["f2"] = np.array([res["f2"] for res in results])

    return metadata


def _get_default_opt(warmstart_configs: Optional[Dict[str, np.ndarray]] = None) -> TPEOptimizer:
    dim, r = 2, 5
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-r, upper=r))

    opt = TPEOptimizer(
        obj_func=lambda eval_config: sphere(eval_config, shift=0.0),
        config_space=cs,
        min_bandwidth_factor=1e-2,
        max_evals=30,
        only_requirements=True,
        metadata=get_metadata(dim),
        warmstart_configs=warmstart_configs,
    )
    return opt


def _get_default_multi_opt() -> TPEOptimizer:
    dim, r = 2, 5
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-r, upper=r))

    opt = TPEOptimizer(
        obj_func=multi_modal_MOP,
        config_space=cs,
        objective_names=["f1", "f2"],
        min_bandwidth_factor=1e-2,
        max_evals=30,
        only_requirements=True,
        metadata=get_mo_metadata(dim),
    )
    return opt


def test_raise_error_in_init() -> None:
    dim, r = 2, 5
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-r, upper=r))

    params = dict(
        obj_func=lambda eval_config: sphere(eval_config, shift=0.0),
        config_space=cs,
        min_bandwidth_factor=1e-2,
        max_evals=30,
        only_requirements=True,
    )
    with pytest.raises(KeyError):
        metadata = get_metadata(dim)
        metadata[OBJECTIVE_KEY] = {}
        TPEOptimizer(metadata=metadata, **params)
    with pytest.raises(NotImplementedError):
        metadata = get_metadata(dim)
        constraints = {}
        constraints["dummy"] = np.inf
        TPEOptimizer(metadata=metadata, constraints=constraints, **params)


@pytest.mark.skip("Feature to update meta-tasks is not ready")  # type: ignore
def test_update_observations() -> None:
    pass


def test_apply_knowledge_augmentation() -> None:
    opt = _get_default_opt()
    ka_configs = get_metadata(2)["shift=1"]
    opt.apply_knowledge_augmentation(observations=ka_configs)


def test_get_task_weights() -> None:
    opt = _get_default_opt()
    opt._max_evals = 100
    opt.optimize()
    assert isinstance(opt._sampler, MetaLearnTPE)
    task_weights = opt._sampler.get_task_weights()
    assert np.isclose(task_weights.sum(), 1.0)


def test_compute_probability_improvement_and_get_config_candidates() -> None:
    opt = _get_default_opt()
    opt.optimize()
    cands = opt._sampler.get_config_candidates()
    assert cands["x0"].size == 24 * 3
    pi = opt._sampler.compute_probability_improvement(cands)
    assert pi.size == 24 * 3


def test_observations() -> None:
    opt = _get_default_opt()
    opt.optimize()
    data = opt.fetch_observations()
    assert all(name in data for name in opt._hp_names)
    assert "loss" in data


def test_optimize() -> None:
    warmstart_configs = get_metadata(2)["shift=1"]
    opt = _get_default_opt(warmstart_configs=warmstart_configs)
    opt.optimize()


def test_optimize_with_warmstart() -> None:
    opt = _get_default_opt()
    opt.optimize()


def test_optimize_with_motpe() -> None:
    opt = _get_default_multi_opt()
    opt.optimize()


if __name__ == "__main__":
    unittest.main()
