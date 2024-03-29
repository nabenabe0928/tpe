[![Build Status](https://github.com/nabenabe0928/tpe/workflows/Functionality%20test/badge.svg?branch=stable)](https://github.com/nabenabe0928/tpe)
[![codecov](https://codecov.io/gh/nabenabe0928/tpe/branch/stable/graph/badge.svg?token=UXC2K5VJNN)](https://codecov.io/gh/nabenabe0928/tpe)

# Introduction
This package is the implementation example of tree-structured parzen estimator (TPE).
TPE is an hyperparameter optimization (HPO) method invented in [`Algorithms for Hyper-Parameter Optimization`](https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf).

**NOTE**: The sampling strategy is based on the [BOHB](http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) implementation.

# Setup
This package requires python 3.8 or later version and you can install 
```
pip install tpe
```

# Running example
The optimization of 10D sphere function can be executed as follows:

```python
from __future__ import annotations

import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.optimizer import TPEOptimizer


def sphere(eval_config: dict[str, float]) -> tuple[dict[str, float], float]:
    start = time.time()
    vals = np.array(list(eval_config.values()))
    vals *= vals
    return {"loss": np.sum(vals)}, time.time() - start


if __name__ == "__main__":
    dim = 10
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-5, upper=5))

    opt = TPEOptimizer(obj_func=sphere, config_space=cs, resultfile='sphere')
    # If you do not want to do logging, remove the `logger_name` argument
    print(opt.optimize(logger_name="sphere"))
```

The documentation of `ConfigSpace` is available [here](https://automl.github.io/ConfigSpace/master/).

# Citation

Please cite the following paper when using my implementation:

```bibtex
@article{watanabe2023tpe,
  title   = {Tree-structured {P}arzen estimator: Understanding its algorithm components and their roles for better empirical performance},
  author  = {S. Watanabe},
  journal = {arXiv preprint arXiv:2304.11127},
  year    = {2023}
}
```
