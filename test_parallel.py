import glob
import hashlib
import os
import time
from typing import Any, Callable, Dict, Protocol

import numpy as np

import ujson as json


DIR_NAME = "scheduler/"
TOKEN_PATTERN = "scheduler_*.token"
PUBLIC_TOKEN_NAME = "scheduler.token"
SCHEDULER_FILE_NAME = "scheduler.json"
RESULT_FILE_NAME = "results.json"


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
def init_scheduler(public_token: str, private_token: str, scheduler_path: str, result_path: str) -> None:
    for path in [scheduler_path, result_path]:
        if not os.path.exists(path):
            with open(path, mode="w") as f:
                json.dump({}, f, indent=4)


@verify_token
def record_runtime(public_token: str, private_token: str, proc_id: str, runtime: float, dir_name: str) -> float:
    path = os.path.join(dir_name, SCHEDULER_FILE_NAME)
    record = json.load(open(path))
    cumtime = record.get(proc_id, 0.0) + runtime
    record[proc_id] = cumtime
    with open(path, mode="w") as f:
        json.dump(record, f, indent=4)

    return cumtime


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
def is_scheduler_ready(public_token: str, private_token: str, n_procs: int, dir_name: str) -> bool:
    path = os.path.join(dir_name, SCHEDULER_FILE_NAME)
    return len(json.load(open(path))) == n_procs


@verify_token
def is_min_cumtime(public_token: str, private_token: str, proc_id: str, dir_name: str) -> bool:
    path = os.path.join(dir_name, SCHEDULER_FILE_NAME)
    cumtimes = json.load(open(path))
    proc_cumtime = cumtimes[proc_id]
    return min(cumtime for cumtime in cumtimes.values()) == proc_cumtime


def wait_all_procs(
    public_token: str, private_token: str, n_procs: int, dir_name: str, waiting_time: float = 5e-3
) -> None:
    start = time.time()
    kwargs = dict(public_token=public_token, private_token=private_token, n_procs=n_procs, dir_name=dir_name)
    while True:
        waiting_time *= 1.02
        if is_scheduler_ready(**kwargs):
            return
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
    def __init__(self, public_token: str, private_token: str, proc_id: str, dir_name: str, func: ObjectiveFunc):
        self._kwargs = dict(public_token=public_token, private_token=private_token, proc_id=proc_id, dir_name=dir_name)
        record_runtime(**self._kwargs, runtime=0.0)
        self._func = func
        self._proc_id = proc_id
        self._prev_timestamp = time.time()

    def __call__(self, eval_config: Dict[str, Any]) -> Dict[str, float]:
        sampling_time = time.time() - self._prev_timestamp
        output = self._func(eval_config)
        loss, runtime = output["loss"], output["runtime"]
        cumtime = record_runtime(**self._kwargs, runtime=runtime+sampling_time)
        wait_until_next(**self._kwargs)
        self._prev_timestamp = time.time()
        row = dict(loss=loss, cumtime=cumtime, proc_id=self._proc_id, timestamp=self._prev_timestamp)
        record_result(**self._kwargs, results=row)
        return output

    def finish(self) -> None:
        inf_time = 1 << 40
        record_runtime(**self._kwargs, runtime=inf_time)


def get_wrapper_func(func: ObjectiveFunc, n_procs: int, subdir_name: str) -> WrapperFunc:
    proc_id = generate_time_hash()
    dir_name = os.path.join(DIR_NAME, subdir_name)
    public_token = os.path.join(dir_name, PUBLIC_TOKEN_NAME)
    private_token = os.path.join(dir_name, f"scheduler_{proc_id}.token")

    os.makedirs(dir_name, exist_ok=True)
    token_pattern = os.path.join(dir_name, TOKEN_PATTERN)
    scheduler_path = os.path.join(dir_name, SCHEDULER_FILE_NAME)
    result_path = os.path.join(dir_name, RESULT_FILE_NAME)
    init_token(token_pattern=token_pattern, public_token=public_token)
    kwargs = dict(public_token=public_token, private_token=private_token)
    init_scheduler(**kwargs, scheduler_path=scheduler_path, result_path=result_path)
    _func = WrapperFunc(**kwargs, func=func, proc_id=proc_id, dir_name=dir_name)
    wait_all_procs(**kwargs, n_procs=n_procs, dir_name=dir_name)
    return _func


if __name__ == "__main__":
    func = get_wrapper_func(
        func=lambda eval_config: dict(loss=eval_config["x"] ** 2, runtime=1.0),
        n_procs=4,
        subdir_name="test"
    )
    for i in range(10):
        func({"x": 0.1 * i})
    else:
        func.finish()
