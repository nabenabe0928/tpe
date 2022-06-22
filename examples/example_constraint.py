import time
from typing import Dict, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import matplotlib.pyplot as plt

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


def plot_result(observations: Dict[str, np.ndarray]) -> None:
    is_feasible = observations["c"] <= -0.95
    # xmin, xmax = loss_vals.min(), loss_vals.max()
    plt.scatter(
        observations["x0"][is_feasible],
        observations["x1"][is_feasible],
        color="blue",
        label="Feasible solutions",
        s=2.0,
    )
    plt.scatter(
        observations["x0"][~is_feasible],
        observations["x1"][~is_feasible],
        color="red",
        label="Infeasible solutions",
        s=2.0,
    )
    plt.xlim(-0.1, 6.1)
    plt.ylim(-0.1, 6.1)
    # plt.scatter(loss_vals, observations["c"], label="Observations")
    # plt.hlines(-0.95, xmin, xmax, label="Threshold")
    plt.title(f'Optimal: {observations["loss"][is_feasible].min()}')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    dim = 2
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=0, upper=6))

    opt = TPEOptimizer(
        obj_func=small2d,
        config_space=cs,
        min_bandwidth_factor=1e-2,
        max_evals=100,
        constraints={"c": -0.95},
    )
    opt.optimize(logger_name="small2d")
    plot_result(opt.fetch_observations())
