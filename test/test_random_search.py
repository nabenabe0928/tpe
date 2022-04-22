import os

from typing import Dict, Tuple, Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import unittest

from tpe.optimizer.random_search import RandomSearch
from tpe.utils.utils import get_logger


def mix_func(eval_config: Dict[str, Union[str, float]]) -> Tuple[Dict[str, float], float]:
    func_dict = {"cosine": np.cos, "sine": np.sin}
    vals = np.array([v for v in eval_config.values() if isinstance(v, float)])
    vals *= vals
    assert isinstance(eval_config["func"], str)
    func_name: str = eval_config["func"]
    return {"loss": 1 - func_dict[func_name](vals.sum()) ** 2}, 0.0


def cleanup() -> None:
    os.remove("log/test.log")
    os.remove("results/test.json")
    os.remove("incumbents/test.json")


class TestRandomSearch(unittest.TestCase):
    def setUp(self) -> None:
        dim = 10
        self.cs = CS.ConfigurationSpace()
        self.cs.add_hyperparameter(CSH.CategoricalHyperparameter("func", choices=["sine", "cosine"]))
        for d in range(dim):
            if d < dim - 1:
                self.cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-5, upper=5))
            else:
                self.cs.add_hyperparameter(
                    CSH.OrdinalHyperparameter(f"x{d}", sequence=list(range(-5, 6)), meta={"lower": -5, "upper": 5})
                )

        self.hp_names_cat = list(self.cs._hyperparameters.keys())
        self.logger = get_logger(file_name="test", logger_name="test")

    def test_optimize(self) -> None:
        max_evals = 100

        opt = RandomSearch(obj_func=mix_func, config_space=self.cs, max_evals=max_evals, resultfile="test")
        opt.optimize(self.logger)
        assert opt.fetch_observations()["x0"].size == max_evals

        cleanup()


if __name__ == "__main__":
    unittest.main()
