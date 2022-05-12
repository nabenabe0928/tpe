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
from typing import Dict

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from tpe.optimizer import TPEOptimizer


def sphere(eval_config: Dict[str, float]) -> float:
    vals = np.array(list(eval_config.values()))
    vals *= vals
    return np.sum(vals)


if __name__ == '__main__':
    dim = 10
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f'x{d}', lower=-5, upper=5))

    opt = TPEOptimizer(obj_func=sphere, config_space=cs, resultfile='sphere')
    # If you do not want to do logging, remove the `logger_name` argument
    opt.optimize(logger_name="sphere")
```
