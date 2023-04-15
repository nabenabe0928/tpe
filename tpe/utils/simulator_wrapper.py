import glob
import hashlib
import os
import time
from typing import Any, Callable, Dict, List, Protocol

import numpy as np

import ujson as json


DIR_NAME = "scheduler/"
TOKEN_PATTERN = "scheduler_*.token"
PUBLIC_TOKEN_NAME = "scheduler.token"
SCHEDULER_FILE_NAME = "scheduler.json"
RESULT_FILE_NAME = "results.json"
PROC_ALLOC_NAME = "proc_alloc.json"
RUNTIME_CACHE_FILE_NAME = "runtime_cache.json"


def generate_time_hash() -> str:
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode("utf-8"))
    return hash.hexdigest()


def publish_token(public_token: str, private_token: str, waiting_time: float = 1e-4) -> None:
    start = time.time()
    waiting_time *= 1 + np.random.random()
    while True:
        try:
            os.rename(public_token, private_token)
            return
        except FileNotFoundError:
            time.sleep(waiting_time)
            if time.time() - start >= 10:
                raise TimeoutError("Timeout during token publication. Please remove token files and try again.")


def remove_token(public_token: str, private_token: str) -> None:
    os.rename(private_token, public_token)


def verify_token(func: Callable) -> Callable:
    def _inner(public_token: str, private_token: str, **kwargs):
        publish_token(public_token=public_token, private_token=private_token)
        output = func(public_token=public_token, private_token=private_token, **kwargs)
        remove_token(public_token=public_token, private_token=private_token)
        return output

    return _inner


def init_token(token_pattern: str, public_token: str) -> None:
    n_tokens = len(glob.glob(token_pattern))
    if n_tokens == 0:
        with open(public_token, mode="w"):
            pass
    elif n_tokens > 1:  # Token from another process could exist!
        raise FileExistsError


@verify_token
def init_scheduler(public_token: str, private_token: str, dir_name: str) -> None:
    for fn in [SCHEDULER_FILE_NAME, RESULT_FILE_NAME, RUNTIME_CACHE_FILE_NAME, PROC_ALLOC_NAME]:
        path = os.path.join(dir_name, fn)
        if not os.path.exists(path):
            with open(path, mode="w") as f:
                json.dump({}, f, indent=4)


@verify_token
def allocate_worker(public_token: str, private_token: str, dir_name: str, pid: int) -> None:
    path = os.path.join(dir_name, PROC_ALLOC_NAME)
    cur_alloc = json.load(open(path))
    cur_alloc[pid] = 0
    with open(path, mode="w") as f:
        json.dump(cur_alloc, f, indent=4)


@verify_token
def complete_worker_allocation(public_token: str, private_token: str, dir_name: str) -> Dict[int, int]:
    path = os.path.join(dir_name, PROC_ALLOC_NAME)
    alloc = json.load(open(path))
    sorted_pids = np.sort([int(pid) for pid in alloc.keys()])
    alloc = {pid: idx for idx, pid in enumerate(sorted_pids)}
    with open(path, mode="w") as f:
        json.dump(alloc, f, indent=4)

    return alloc


@verify_token
def record_cumtime(public_token: str, private_token: str, proc_id: str, runtime: float, dir_name: str) -> float:
    path = os.path.join(dir_name, SCHEDULER_FILE_NAME)
    record = json.load(open(path))
    cumtime = record.get(proc_id, 0.0) + runtime
    record[proc_id] = cumtime
    with open(path, mode="w") as f:
        json.dump(record, f, indent=4)

    return cumtime


@verify_token
def cache_runtime(
    public_token: str, private_token: str, config_key: str, runtime: float, dir_name: str, update: bool = True
) -> None:
    path = os.path.join(dir_name, RUNTIME_CACHE_FILE_NAME)
    cache = json.load(open(path))
    if config_key not in cache:
        cache[config_key] = [runtime]
    elif update:
        cache[config_key][0] = runtime
    else:
        cache[config_key].append(runtime)

    cache[config_key] = np.sort(cache[config_key]).tolist()
    with open(path, mode="w") as f:
        json.dump(cache, f, indent=4)


@verify_token
def delete_runtime(public_token: str, private_token: str, config_key: str, index: float, dir_name: str) -> None:
    path = os.path.join(dir_name, RUNTIME_CACHE_FILE_NAME)
    cache = json.load(open(path))
    n_configs = len(cache.get(config_key, [])) > 0
    if n_configs <= 1:
        cache[config_key] = [0.0]  # we need to have at least one element.
    else:
        cache[config_key].pop(index)

    with open(path, mode="w") as f:
        json.dump(cache, f, indent=4)


@verify_token
def fetch_cache_runtime(public_token: str, private_token: str, dir_name: str) -> None:
    path = os.path.join(dir_name, RUNTIME_CACHE_FILE_NAME)
    return json.load(open(path))


@verify_token
def record_result(
    public_token: str, private_token: str, proc_id: str, results: Dict[str, float], dir_name: str
) -> None:
    path = os.path.join(dir_name, RESULT_FILE_NAME)
    record = json.load(open(path))
    for key, val in results.items():
        if key not in record:
            record[key] = [val]
        else:
            record[key].append(val)

    with open(path, mode="w") as f:
        json.dump(record, f, indent=4)


@verify_token
def is_terminated(public_token: str, private_token: str, dir_name: str, max_evals: int) -> bool:
    path = os.path.join(dir_name, RESULT_FILE_NAME)
    return len(json.load(open(path))["loss"]) >= max_evals


@verify_token
def is_scheduler_ready(public_token: str, private_token: str, n_procs: int, dir_name: str) -> bool:
    path = os.path.join(dir_name, SCHEDULER_FILE_NAME)
    return len(json.load(open(path))) == n_procs


@verify_token
def is_allocation_ready(public_token: str, private_token: str, n_procs: int, dir_name: str) -> bool:
    path = os.path.join(dir_name, PROC_ALLOC_NAME)
    return len(json.load(open(path))) == n_procs


@verify_token
def get_proc_id_to_idx(public_token: str, private_token: str, dir_name: str) -> Dict[str, int]:
    path = os.path.join(dir_name, SCHEDULER_FILE_NAME)
    return {k: idx for idx, k in enumerate(json.load(open(path)).keys())}


@verify_token
def is_min_cumtime(public_token: str, private_token: str, proc_id: str, dir_name: str) -> bool:
    path = os.path.join(dir_name, SCHEDULER_FILE_NAME)
    cumtimes = json.load(open(path))
    proc_cumtime = cumtimes[proc_id]
    return min(cumtime for cumtime in cumtimes.values()) == proc_cumtime


def wait_worker_allocation(
    public_token: str, private_token: str, n_procs: int, dir_name: str, waiting_time: float = 1e-2
) -> Dict[int, int]:
    start = time.time()
    kwargs = dict(public_token=public_token, private_token=private_token, dir_name=dir_name)
    waiting_time *= 1 + np.random.random()
    while True:
        if is_allocation_ready(**kwargs, n_procs=n_procs):
            return complete_worker_allocation(**kwargs)
        else:
            time.sleep(waiting_time)
            if time.time() - start >= 5:
                raise TimeoutError("Timeout in the allocation of workers. Please make sure n_procs is correct.")


def wait_all_procs(
    public_token: str, private_token: str, n_procs: int, dir_name: str, waiting_time: float = 1e-2
) -> Dict[str, int]:
    start = time.time()
    kwargs = dict(public_token=public_token, private_token=private_token, dir_name=dir_name)
    waiting_time *= 1 + np.random.random()
    while True:
        if is_scheduler_ready(**kwargs, n_procs=n_procs):
            return get_proc_id_to_idx(**kwargs)
        else:
            time.sleep(waiting_time)
            if time.time() - start >= 5:
                raise TimeoutError("Timeout in creating a scheduler. Please make sure n_procs is correct.")


def wait_until_next(
    public_token: str, private_token: str, proc_id: str, dir_name: str, waiting_time: float = 1e-4
) -> None:
    waiting_time *= 1 + np.random.random()
    kwargs = dict(public_token=public_token, private_token=private_token, proc_id=proc_id, dir_name=dir_name)
    while True:
        if is_min_cumtime(**kwargs):
            return
        else:
            time.sleep(waiting_time)


class ObjectiveFunc(Protocol):
    def __call__(self, eval_config: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError


class WrapperFunc:
    def __init__(
        self,
        public_token: str,
        private_token: str,
        proc_id: str,
        dir_name: str,
        n_procs: int,
        func: ObjectiveFunc,
        max_budget: int,
        runtime_key: str = "runtime",
        loss_key: str = "loss",
    ):
        self._kwargs = dict(public_token=public_token, private_token=private_token, dir_name=dir_name)
        record_cumtime(**self._kwargs, proc_id=proc_id, runtime=0.0)
        self._func = func
        self._max_budget = max_budget
        self._runtime_key = runtime_key
        self._loss_key = loss_key
        self._proc_id = proc_id
        self._proc_id_to_index = wait_all_procs(
            public_token=public_token, private_token=private_token, n_procs=n_procs, dir_name=dir_name
        )
        self._index = self._proc_id_to_index[self._proc_id]
        time.sleep(1e-2)  # buffer before the optimization
        self._prev_timestamp = time.time()

    def _get_cached_runtime_index(self, cached_runtimes: List[float], config_key: str, runtime: float) -> int:
        # a[i-1] < v <= a[i]: np.searchsorted(..., side="left")
        idx = np.searchsorted(cached_runtimes, runtime, side="left")
        return max(0, idx - 1)

    def _proc_output(self, eval_config: Dict[str, Any], budget: int) -> Dict[str, float]:
        output = self._func(eval_config, budget)
        config_key = str(eval_config)
        loss, total_runtime = output[self._loss_key], output[self._runtime_key]
        cached_runtimes = fetch_cache_runtime(**self._kwargs).get(config_key, [0.0])
        cached_runtime_index = self._get_cached_runtime_index(cached_runtimes, config_key, total_runtime)
        cached_runtime = cached_runtimes[cached_runtime_index]

        actual_runtime = max(0.0, total_runtime - cached_runtime)
        # Start from the intermediate result, and hence we overwrite the cached runtime
        overwrite_min_runtime = cached_runtime < total_runtime
        if budget != self._max_budget:  # update the cache data
            cache_runtime(**self._kwargs, config_key=config_key, runtime=total_runtime, update=overwrite_min_runtime)
        else:
            delete_runtime(**self._kwargs, config_key=config_key, index=cached_runtime_index)

        return {self._loss_key: loss, self._runtime_key: actual_runtime}

    def __call__(self, eval_config: Dict[str, Any], budget: int) -> Dict[str, float]:
        sampling_time = time.time() - self._prev_timestamp
        output = self._proc_output(eval_config, budget)
        loss, runtime = output[self._loss_key], output[self._runtime_key]
        cumtime = record_cumtime(**self._kwargs, proc_id=self._proc_id, runtime=runtime+sampling_time)
        wait_until_next(**self._kwargs, proc_id=self._proc_id)
        self._prev_timestamp = time.time()
        row = dict(loss=loss, cumtime=cumtime, proc_index=self._index)
        record_result(**self._kwargs, proc_id=self._proc_id, results=row)
        return output

    def finish(self) -> None:
        inf_time = 1 << 40
        record_cumtime(**self._kwargs, proc_id=self._proc_id, runtime=inf_time)


def get_wrapper_func(
    func: ObjectiveFunc,
    n_procs: int,
    subdir_name: str,
    max_budget: int,
    runtime_key: str = "runtime",
    loss_key: str = "loss",
) -> WrapperFunc:
    proc_id = generate_time_hash()
    dir_name = os.path.join(DIR_NAME, subdir_name)
    public_token = os.path.join(dir_name, PUBLIC_TOKEN_NAME)
    private_token = os.path.join(dir_name, f"scheduler_{proc_id}.token")
    os.makedirs(dir_name, exist_ok=True)
    token_pattern = os.path.join(dir_name, TOKEN_PATTERN)
    init_token(token_pattern=token_pattern, public_token=public_token)
    kwargs = dict(public_token=public_token, private_token=private_token, dir_name=dir_name)
    init_scheduler(**kwargs)
    _func = WrapperFunc(
        **kwargs,
        func=func,
        proc_id=proc_id,
        n_procs=n_procs,
        max_budget=max_budget,
        runtime_key=runtime_key,
        loss_key=loss_key,
    )
    return _func
