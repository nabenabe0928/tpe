from typing import Dict, List

import json
import os


def save_observations(dir_name: str, file_name: str, data: Dict[str, List[List[float]]]) -> None:
    path = os.path.join("results", dir_name)
    with open(os.path.join(path, file_name), mode="w") as f:
        json.dump(data, f, indent=4)


def exist_file(dir_name: str, file_name: str) -> bool:
    path = os.path.join("results", dir_name)
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        print(f"{file_path} exists; skip")
        return True
    else:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, file_name), mode="w"):
            pass

        return False
