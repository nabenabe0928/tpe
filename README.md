# Introduction
This package is the implementation example of tree-structured parzen estimator (TPE).
TPE is an hyperparameter optimization (HPO) method invented in [`Algorithms for Hyper-Parameter Optimization`](https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf).

**NOTE**: The sampling strategy is based on the [BOHB](http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) implementation.

# Setup
This package requires python 3.7 or later version and you can install 
```
pip install -r requirements.txt
```

# Running example
The example using `sphere function` is provided in `run_tpe.py` and you can run by `python run_tpe.py`.
