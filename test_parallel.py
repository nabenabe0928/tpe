import glob
import hashlib
import os
import time
from typing import Callable, Dict

import ujson as json


DIR_NAME = "scheduler/"
TOKEN_PATTERN = "scheduler_*.token"
PUBLIC_TOKEN_NAME = "scheduler.token"
SCHEDULER_FILE_NAME = "scheduler.json"


def generate_time_hash() -> str:
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode("utf-8"))
    return hash.hexdigest()


def publish_token(public_token: str, private_token: str, waiting_time: float = 1e-5) -> None:
    start = time.time()
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
def init_scheduler(public_token: str, private_token: str, scheduler_path: str) -> None:
    if not os.path.exists(scheduler_path):
        with open(scheduler_path, mode="w") as f:
            json.dump({}, f, indent=4)


@verify_token
def record_runtime(public_token: str, private_token: str, proc_id: str, runtime: float, dir_name: str) -> float:
    path = os.path.join(dir_name, SCHEDULER_FILE_NAME)
    old = json.load(open(path))
    new = old.copy()
    cumtime = new.get(proc_id, 0.0) + runtime
    new[proc_id] = cumtime
    with open(path, mode="w") as f:
        json.dump(new, f, indent=4)

    return cumtime


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
    public_token: str, private_token: str, n_procs: int, dir_name: str, waiting_time: float = 1e-5
) -> None:
    kwargs = dict(public_token=public_token, private_token=private_token, n_procs=n_procs, dir_name=dir_name)
    while True:
        if is_scheduler_ready(**kwargs):
            return
        else:
            time.sleep(waiting_time)


def wait_until_next(
    public_token: str, private_token: str, proc_id: str, dir_name: str, waiting_time: float = 1e-5
) -> None:
    kwargs = dict(public_token=public_token, private_token=private_token, proc_id=proc_id, dir_name=dir_name)
    while True:
        if is_min_cumtime(**kwargs):
            return
        else:
            time.sleep(waiting_time)


def get_wrapper_func(
    public_token: str, private_token: str, proc_id: str, wrapper_func: Callable, n_procs: int, dir_name: str
) -> Callable:
    os.makedirs(dir_name, exist_ok=True)
    token_pattern = os.path.join(dir_name, TOKEN_PATTERN)
    scheduler_path = os.path.join(dir_name, SCHEDULER_FILE_NAME)
    init_token(token_pattern=token_pattern, public_token=public_token)
    init_scheduler(public_token=public_token, private_token=private_token, scheduler_path=scheduler_path)
    func = wrapper_func(public_token=public_token, private_token=private_token, proc_id=proc_id, dir_name=dir_name)
    wait_all_procs(public_token=public_token, private_token=private_token, n_procs=n_procs, dir_name=dir_name)
    return func


def wrapper_func(public_token: str, private_token: str, proc_id: str, dir_name: str) -> Callable:
    print(proc_id)
    kwargs = dict(public_token=public_token, private_token=private_token, proc_id=proc_id, dir_name=dir_name)
    record_runtime(**kwargs, runtime=0.0)

    def _func(x: float) -> Dict[str, float]:
        start = time.time()
        loss_val = x ** 2
        runtime = time.time() - start
        cumtime = record_runtime(**kwargs, runtime=runtime)
        wait_until_next(**kwargs)
        print(proc_id, runtime, cumtime)
        return dict(loss=loss_val, runtime=runtime)

    return _func


if __name__ == "__main__":
    proc_id = generate_time_hash()
    dir_name = os.path.join(DIR_NAME, "test")
    kwargs = dict(
        proc_id=proc_id,
        dir_name=dir_name,
        public_token=os.path.join(dir_name, PUBLIC_TOKEN_NAME),
        private_token=os.path.join(dir_name, f"scheduler_{proc_id}.token"),
    )
    func = get_wrapper_func(wrapper_func=wrapper_func, n_procs=4, **kwargs)

    for i in range(10):
        func(0.1 * i)
    else:
        inf_time = 1 << 40
        record_runtime(**kwargs, runtime=inf_time)
