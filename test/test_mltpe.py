import unittest
from typing import Dict, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.optimizer import TPEOptimizer


def sphere(eval_config: Dict[str, float], shift: float = 0.0) -> Tuple[Dict[str, float], float]:
    vals = np.array(list(eval_config.values())) - shift
    vals *= vals
    return {"loss": vals.sum()}, 0.0


def get_metadata(dim: int) -> Dict[str, Dict[str, np.ndarray]]:
    n_tasks, n_evals = 3, 50
    metadata: Dict[str, Dict[str, np.ndarray]] = {
        f"shift=0.{i}": {f"x{d}": np.random.random(n_evals) for d in range(dim)} for i in range(1, n_tasks)
    }
    for task_name in metadata.keys():
        metadata[task_name]["loss"] = np.array(
            [sphere({f"x{d}": metadata[task_name][f"x{d}"][i] for d in range(dim)})[0]["loss"] for i in range(n_evals)]
        )

    return metadata


def _get_default_opt() -> TPEOptimizer:
    dim, r = 2, 5
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-r, upper=r))

    opt = TPEOptimizer(
        obj_func=lambda eval_config: sphere(eval_config, shift=0.1),
        config_space=cs,
        min_bandwidth_factor=1e-2,
        max_evals=30,
        only_requirements=True,
        metadata=get_metadata(dim),
    )
    return opt


def test_raise_error_in_init() -> None:
    pass


def test_init_samplers() -> None:
    pass


def test_update_observations() -> None:
    pass


def test_apply_knowledge_augmentation() -> None:
    pass


def test_compute_task_similarity() -> None:
    pass


def test_compute_task_weights() -> None:
    pass


def test_compute_probability_improvement() -> None:
    pass


def test_get_config_candidates() -> None:
    pass


def test_observations() -> None:
    pass


def test_optimize() -> None:
    opt = _get_default_opt()
    opt.optimize()


if __name__ == "__main__":
    unittest.main()
