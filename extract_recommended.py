import os

import numpy as np

import pandas as pd

import ujson as json


DIR_NAME = "results/recommended/"


if __name__ == "__main__":
    files = os.listdir(DIR_NAME)
    mean_vals = {"target": []}
    mean_vals.update({f"n_evals{i+1:0>3}": [] for i in range(200)})

    for fn in files:
        cum_vals = np.minimum.accumulate(json.load(open(os.path.join(DIR_NAME, fn))), axis=-1)
        means = np.mean(cum_vals, axis=0)
        mean_vals["target"].append(fn.split(".json")[0])

        for i in range(200):
            mean_vals[f"n_evals{i+1:0>3}"].append(means[i])

    mean_vals = {k: np.asarray(v, dtype=np.float32) if k.startswith("n_evals") else v for k, v in mean_vals.items()}
    pd.DataFrame(mean_vals).to_csv("results/recommended.csv", index=False)
