from typing import Dict, List

import json
import os


DEFAULT_RESULT_DIR = "results"


def save_observations(
    dir_name: str,
    file_name: str,
    data: Dict[str, List[List[float]]],
    result_dir: str = DEFAULT_RESULT_DIR,
) -> None:
    path = os.path.join(result_dir, dir_name)
    with open(os.path.join(path, file_name), mode="w") as f:
        json.dump(data, f, indent=4)


def exist_file(
    dir_name: str,
    file_name: str,
    result_dir: str = DEFAULT_RESULT_DIR,
) -> bool:
    path = os.path.join(result_dir, dir_name)
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        print(f"{file_path} exists; skip")
        return True
    else:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, file_name), mode="w"):
            pass

        return False
