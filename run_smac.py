from typing import Any, Dict, Optional

import ConfigSpace as CS

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband

from tpe.utils.simulator_wrapper import CentralWorker


class CentralWorkerForSMAC(CentralWorker):
    def __call__(self, eval_config: Dict[str, Any], budget: int, seed: Optional[int] = None) -> float:
        output = super().__call__(eval_config, budget)
        return output[self._loss_key]


def obj_func(eval_config: CS.Configuration, budget: int) -> float:
    print(eval_config)
    return dict(loss=eval_config["x"] ** 2, runtime=budget / 10.0)


if __name__ == "__main__":
    max_budget = 100
    max_evals = 100
    n_workers = 8

    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5, 5))
    scenario = Scenario(
        config_space,
        n_trials=max_evals,
        min_budget=1,
        max_budget=max_budget,
        n_workers=n_workers,
    )
    print("Start")
    worker = CentralWorkerForSMAC(
        obj_func=obj_func,
        n_workers=n_workers,
        max_budget=max_budget,
        max_evals=max_evals - n_workers + 1,
        subdir_name="smac",
        loss_key="loss",
        runtime_key="runtime"
    )
    print("Finish")
    smac = MFFacade(
        scenario,
        worker,
        initial_design=MFFacade.get_initial_design(scenario, n_configs=5),
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
        overwrite=True,
    )
    incumbent = smac.optimize()
    print("Finish")
