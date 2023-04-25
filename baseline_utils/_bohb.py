import time
from multiprocessing import Pool
from typing import Dict, List

import ConfigSpace as CS

from hpbandster.core import nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

from tpe.utils.simulator_wrapper import WorkerFunc, is_simulator_terminated
from tpe.utils.utils import AbstractObjFunc


LOSS_KEY = "loss"
RUNTIME_KEY = "runtime"


class BOHBWorker(Worker):
    def __init__(
        self,
        worker: WorkerFunc,
        max_evals: int,
        sleep_interval: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sleep_interval = sleep_interval
        self._worker = worker
        self._max_evals = max_evals

    def compute(self, config: Dict, budget: float, **kwargs):
        output = self._worker(config, budget)
        if is_simulator_terminated(self._worker._result_path, max_evals=self._max_evals):
            self._worker.finish()  # after the termination, no record will happen

        return dict(loss=output["loss"])


def get_bohb_workers(
    obj_func: AbstractObjFunc,
    subdir_name: str,
    n_workers: int,
    max_evals: int,
    max_budget: int,
    run_id: str,
    ns_host: str,
    loss_key: str = LOSS_KEY,
    runtime_key: str = RUNTIME_KEY,
) -> List[WorkerFunc]:
    kwargs = dict(
        func=obj_func,
        n_workers=n_workers,
        subdir_name=subdir_name,
        max_budget=max_budget,
        loss_key=loss_key,
        runtime_key=runtime_key,
    )

    pool = Pool()
    results = []
    for _ in range(n_workers):
        time.sleep(5e-3)
        results.append(pool.apply_async(WorkerFunc, kwds=kwargs))

    pool.close()
    pool.join()

    workers = [result.get() for result in results]
    bohb_workers = []
    kwargs = dict(sleep_interval=0.5, nameserver=ns_host, run_id=run_id)
    for i in range(n_workers):
        worker = BOHBWorker(worker=workers[i], max_evals=max_evals, id=i, **kwargs)
        worker.run(background=True)
        bohb_workers.append(worker)

    return bohb_workers


def run_bohb(
    obj_func: AbstractObjFunc,
    config_space: CS.ConfigurationSpace,
    min_budget: int,
    max_budget: int,
    run_id: str,
    n_workers: int,
    subdir_name: str,
    ns_host: str = "127.0.0.1",
    max_evals: int = 450,  # eta=3,S=2,100 full evals
    n_brackets: int = 70,  # 22 HB iter --> 33 SH brackets
) -> None:
    ns = hpns.NameServer(run_id=run_id, host=ns_host, port=None)
    ns.start()
    _ = get_bohb_workers(
        obj_func=obj_func,
        subdir_name=subdir_name,
        n_workers=n_workers,
        max_evals=max_evals,
        max_budget=max_budget,
        run_id=run_id,
        ns_host=ns_host,
    )
    bohb = BOHB(
        configspace=config_space,
        run_id=run_id,
        min_budget=min_budget,
        max_budget=max_budget,
    )
    bohb.run(n_iterations=n_brackets, min_n_workers=n_workers)
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()
