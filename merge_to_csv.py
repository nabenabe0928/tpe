import os

import numpy as np

import pandas as pd

import ujson as json


if __name__ == "__main__":
    dir_names = os.listdir("results/")
    data = {
        "multivariate": [],
        "quantile": [],
        "alpha": [],
        "weight": [],
        "min_bandwidth_factor": [],
        "min_bandwidth_factor_for_discrete": [],
    }
    epochs = [epoch - 1 for epoch in [50, 100, 150, 200]]
    results = {
        "setting_index": [],
        "target": [],
        "mean@n_evals050": [],
        "mean@n_evals100": [],
        "mean@n_evals150": [],
        "mean@n_evals200": [],
        "ste@n_evals050": [],
        "ste@n_evals100": [],
        "ste@n_evals150": [],
        "ste@n_evals200": [],
    }
    cols = list(data.keys())

    index = 0
    for dir_name in dir_names:
        for col in cols:
            if col not in dir_name:
                data[col].append(None)
                continue

            val = dir_name[dir_name.find(col):].split("=")[1].split("_")[0]
            data[col].append(val)
        else:
            dir_path = os.path.join("results", dir_name)
            files = os.listdir(dir_path)
            for fn in files:
                try:
                    cum_vals = np.minimum.accumulate(json.load(open(os.path.join(dir_path, fn))), axis=-1)[:, epochs]
                    means = np.mean(cum_vals, axis=0)
                    stes = np.std(cum_vals, axis=0) / np.sqrt(cum_vals.shape[0])
                    results["setting_index"].append(index)
                    results["target"].append(fn.split(".json")[0])
                    for i in range(4):
                        e = 50 * (i + 1)
                        results[f"mean@n_evals{e:0>3}"].append(means[i])
                        results[f"ste@n_evals{e:0>3}"].append(stes[i])
                except Exception:
                    pass

        index += 1

    print(pd.DataFrame(data).to_csv())
    print(pd.DataFrame(results).to_csv(index=False))
