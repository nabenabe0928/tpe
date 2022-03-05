# Introduction
This package is the implementation example of tree-structured parzen estimator (TPE).
TPE is an hyperparameter optimization (HPO) method invented in [`Algorithms for Hyper-Parameter Optimization`](https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf).

**NOTE**: The sampling strategy is based on the [BOHB](http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) implementation.

# Setup
This package requires python 3.8 or later version and you can install 
```
pip install -r requirements.txt
```

# Running example
The optimization of 10D sphere function can be executed as follows:
```
# Optimize 10D sphere function
$ python optimize_sphere.py

# Optimize the hyperparameters defined in `targets/cnn/hyperparameters.py` and `targets/cnn/params.json`
$ python optimize_cnn.py
```

If you would like to play around using tabular benchmarks, you can download from:
|Benchmark name |Running file|
|:--|:--:|
|[HPOLib](https://github.com/automl/nas_benchmarks)| `optimize_hpolib.py` |
|[NASBench101](https://github.com/nabenabe0928/nasbench)| `optimize_nasbench101.py` |
|[NASBench201](https://github.com/D-X-Y/NATS-Bench)|`optimize_nasbench201.py`|

Note that the searching space for NASBench101 follows [here](https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py) and the path of each dataset must be specified as an argument `path` for each module.

The downloads can be executed as follows:
```
# Create a directory for tabular datasets
$ mkdir ~/tabular_datasets
$ cd ~/tabular_datasets

# The download of HPOLib
$ cd ~/tabular_datasets
$ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
$ tar xf fcnet_tabular_benchmarks.tar.gz

# The download of NASBench101 (Quicker version)
$ cd ~/tabular_datasets
# Table data (0.5GB) with only 1 type of budget
$ wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord

# Table data (2GB) with 4 types of budgets
$ wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord

# The download of NASBench201
$ cd ~/tabular_datasets
$ wget https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view
```

You need to run the following commands to setup NASBench environments:
```
# Quicker version of NASBench101 (plus, tensorflow2+ works)
$ git clone https://github.com/nabenabe0928/nasbench
$ cd nasbench
$ pip install -e .
$ cd ..
$ pip install ./nasbench

# Setup of NASBench201
$ pip install nats_bench
``` 

The run for each experiment can be performed as follows:
```
# Optimize the hyperparameters defined in `targets/hpolib/hyperparameters.py` and `targets/hpolib/params.json`
$ python optimize_hpolib.py

# Optimize the hyperparameters defined in `targets/nasbench101/cifar10B/hyperparameters.py` and `targets/nasbench101/cifar10B/params.json`
# The choices of search_space are {cifar10A, cifar10B, cifar10C}
$ python optimize_nasbench101.py --search_space cifar10B

# Optimize the hyperparameters defined in `targets/nasbench201/hyperparameters.py` and `targets/nasbench201/params.json`
$ python optimize_nasbench201.py
```
