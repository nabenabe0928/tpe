import time
from typing import Dict, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import matplotlib.pyplot as plt

import numpy as np

from tpe.optimizer import TPEOptimizer


DIM = 2


def multi_modal_MOP(config: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    multi-modal MOP

    Optimal at:
        f1 = x1
        f2 = 1 - sqrt(f1)
    """
    start = time.time()
    X = np.array([config[f"x{d}"] for d in range(DIM)])
    f1 = X[0]
    g = 1 + 10 * (DIM - 1) + np.sum(X[1:] ** 2 - 10 * np.cos(2 * np.pi * X[1:]))
    h = 1 - np.sqrt(f1 / g) if f1 <= g else 0
    return {"f1": f1, "f2": g * h}, time.time() - start


def plot_result(observations: Dict[str, float]) -> None:
    ans_f1 = np.linspace(0, 1, 100)
    ans_f2 = 1 - np.sqrt(ans_f1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(observations["f1"], observations["f2"], color="blue", label="Observations")
    plt.scatter(ans_f1, ans_f2, color="red", s=1.0, label="Pareto optimal")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("x0", 0, 1))
    V = 0.5  # 30
    for d in range(1, DIM):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-V, upper=V))

    opt = TPEOptimizer(
        obj_func=multi_modal_MOP,
        config_space=cs,
        objective_names=["f1", "f2"],
        min_bandwidth_factor=1e-3,
        n_init=10,
        max_evals=400,
    )
    opt.optimize(logger_name="MOP")
    plot_result(opt.fetch_observations())
