# Introduction
This package is the implementation example of tree-structured parzen estimator (TPE).
TPE is an hyperparameter optimization (HPO) method invented in [`Algorithms for Hyper-Parameter Optimization`](https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf).

**NOTE**: The sampling strategy is based on the [BOHB](http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) implementation.

# Setup
This package requires python 3.7 or later version and you can install 
```
pip install -r requirements.txt
```

If you would like to play around using a tabular benchmark, you can download [the tabular data for HPO on 4 different datasets](https://github.com/automl/nas_benchmarks):
```
$ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
$ tar xf fcnet_tabular_benchmarks.tar.gz

# Run the optimization using TPE
$ optimize_hpolib.py
```
Note that you need to move the downloaded dataset accordingly or specify the path in `optimize_hpolib.py`.

# Running example
```
# Optimize 10D sphere function
$ python optimize_sphere.py

# Optimize the hyperparameters defined in `cnn/hyperparameters.py` and `cnn/params.json`
$ python optimize_cnn.py

# Optimize the hyperparameters defined in `hpolib/hyperparameters.py` and `hpolib/params.json`
$ python optimize_hpolib.py
```
