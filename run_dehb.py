import os
from multiprocessing import Pool
from typing import Dict

import ConfigSpace as CS

from dehb import DEHB

from tpe.utils.simulator_wrapper import allocate_worker, get_wrapper_func, is_terminated, wait_worker_allocation


def obj_func(config: Dict, budget: float) -> Dict:
    res = {
        "fitness": config["x"] ** 2,
        "cost": budget / 10.0,
    }
    return res


class CentralWorker:
    def __init__(self, n_procs: int):
        kwargs = dict(
            func=obj_func,
            n_procs=n_procs,
            subdir_name="dehb",
            max_budget=100,
            loss_key="fitness",
            runtime_key="cost",
        )
        pool = Pool()
        results = [pool.apply_async(get_wrapper_func, kwds=kwargs) for _ in range(n_procs)]
        pool.close()
        pool.join()
        self._workers = [result.get() for result in results]
        self._dir_name = self._workers[0]._kwargs["dir_name"]
        self._public_token = self._workers[0]._kwargs["public_token"]
        self._n_procs = n_procs
        self._pid_to_index: Dict[int, int] = {}

    def _token_verification_kwawrgs(self, pid: int) -> Dict[str, str]:
        private_token = f"_{pid}.".join(self._public_token.split("."))
        kwargs = dict(public_token=self._public_token, private_token=private_token, dir_name=self._dir_name)
        return kwargs

    def _init_alloc(self, pid: int) -> None:
        kwargs = self._token_verification_kwawrgs(pid)
        allocate_worker(**kwargs, pid=pid)
        self._pid_to_index = wait_worker_allocation(**kwargs, n_procs=self._n_procs)

    def __call__(self, config: Dict, budget: float) -> Dict:
        pid = os.getpid()
        if len(self._pid_to_index) != self._n_procs:
            self._init_alloc(pid)

        worker_index = self._pid_to_index[pid]
        output = self._workers[worker_index](config.get_dictionary(), budget)
        kwargs = self._token_verification_kwawrgs(pid)
        if is_terminated(**kwargs, max_evals=100):
            self._workers[worker_index].finish()

        return output


if __name__ == "__main__":
    n_procs = 4
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x", -5, 5))
    worker = CentralWorker(n_procs=n_procs)

    dehb = DEHB(
        f=worker,
        cs=config_space,
        dimensions=1,
        min_budget=1,
        max_budget=100,
        eta=3,
        client=None,
        n_workers=n_procs,
        output_path="dehb-log/"
    )
    dehb.run(fevals=100)
