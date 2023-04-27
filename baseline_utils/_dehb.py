import ConfigSpace as CS

from dehb import DEHB

from tpe.utils.simulator_wrapper import CentralWorker
from tpe.utils.utils import AbstractObjFunc


def run_dehb(
    obj_func: AbstractObjFunc,
    config_space: CS.ConfigurationSpace,
    min_budget: int,
    max_budget: int,
    n_workers: int,
    subdir_name: str,
    max_evals: int = 450,  # eta=3,S=2,100 full evals
) -> None:
    worker = CentralWorker(
        obj_func=obj_func,
        n_workers=n_workers,
        max_budget=max_budget,
        max_evals=max_evals,
        subdir_name=subdir_name,
        loss_key="fitness",
        runtime_key="cost",
    )

    dehb = DEHB(
        f=worker,
        cs=config_space,
        dimensions=len(config_space),
        min_budget=min_budget,
        max_budget=max_budget,
        eta=3,
        client=None,
        n_workers=n_workers,
        output_path="dehb-log/"
    )
    kwargs = obj_func.get_shared_data()
    dehb.run(fevals=max_evals, **kwargs)
