from typing import Any, Dict, Optional

import ConfigSpace as CS

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband

from tpe.utils.simulator_wrapper import CentralWorker
from tpe.utils.utils import AbstractObjFunc


class CentralWorkerForSMAC(CentralWorker):
    def __call__(
        self, eval_config: Dict[str, Any], budget: int, seed: Optional[int] = None, shared_data=None,
    ) -> float:
        output = super().__call__(eval_config, budget, bench_data=shared_data)
        return output[self._loss_key]


def run_smac(
    obj_func: AbstractObjFunc,
    config_space: CS.ConfigurationSpace,
    min_budget: int,
    max_budget: int,
    n_workers: int,
    subdir_name: str,
    n_init: int,
    max_evals: int = 450,  # eta=3,S=2,100 full evals
) -> None:
    scenario = Scenario(
        config_space,
        n_trials=max_evals + n_workers + 5,
        min_budget=min_budget,
        max_budget=max_budget,
        n_workers=n_workers,
    )
    worker = CentralWorkerForSMAC(
        obj_func=obj_func,
        n_workers=n_workers,
        max_budget=max_budget,
        max_evals=max_evals,
        subdir_name=subdir_name,
        loss_key="loss",
        runtime_key="runtime"
    )
    smac = MFFacade(
        scenario,
        worker,
        initial_design=MFFacade.get_initial_design(scenario, n_configs=n_init),
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
        overwrite=True,
    )
    kwargs = obj_func.get_shared_data()
    smac.optimize(**kwargs)
