import time
from multiprocessing import Pool
from typing import Any, Callable, Dict, List

import ConfigSpace as CS

from hpbandster.core import nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

from tpe.utils.simulator_wrapper import WorkerFunc


class ObjFunc(Worker):
    def __init__(self, worker: WorkerFunc, sleep_interval: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.sleep_interval = sleep_interval
        self._worker = worker
        self._counter = 1000  # change here depending on the budget

    def compute(self, config: Dict, budget: float, **kwargs):
        output = self._worker(config, budget)
        self._counter -= 1
        if self._counter <= 0:
            self._worker.finish()  # after the termination, no record will happen

        return dict(loss=output["loss"])

    @staticmethod
    def get_configspace() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5, 5))
        return config_space


def get_workers(
    obj_func: Callable,
    subdir_name: str,
    n_workers: int,
    max_budget: int,
    loss_key: str,
    runtime_key: str,
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

    return [result.get() for result in results]


def obj_func(eval_config: Dict[str, Any], budget: int) -> Dict[str, float]:
    return dict(loss=eval_config["x"]**2, runtime=budget / 10.0)


n_workers = 4
subdir_name = "bohb"
max_budget = 100
max_evals = 1
loss_key = "loss"
runtime_key = "runtime"
run_id = "test-bohb"
ns_host = "127.0.0.1"
ns = hpns.NameServer(run_id=run_id, host=ns_host, port=None)
ns.start()

workers = get_workers(
    obj_func,
    subdir_name=subdir_name,
    n_workers=n_workers,
    max_budget=max_budget,
    loss_key=loss_key,
    runtime_key=runtime_key,
)
bohb_workers = []
for i in range(n_workers):
    worker = ObjFunc(worker=workers[i], sleep_interval=0.5, nameserver=ns_host, run_id=run_id, id=i)
    worker.run(background=True)
    bohb_workers.append(worker)

bohb = BOHB(
    configspace=ObjFunc.get_configspace(),
    run_id=run_id,
    min_budget=1,
    max_budget=max_budget,
)
bohb.run(n_iterations=max_evals, min_n_workers=n_workers)
bohb.shutdown(shutdown_workers=True)
ns.shutdown()
