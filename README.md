# The basement for the experiments for hyperparameter optimization (HPO)

## Requirements
・python3.7 (3.7.4)

・ConfigSpace 0.4.10 [ (github)](https://github.com/automl/ConfigSpace)

・Pytorch 1.2.0 [ (github)](https://github.com/pytorch/pytorch)

・botorch 0.1.3 [ (github)](https://github.com/pytorch/botorch)

```
pip install ConfigSpace
pip install torch torchvision
pip install botorch
pip install -r requirements.txt
```

## Implementation
An easy example of `main.py`.
Note that the optimization is always minimization;
Therefore, users have to set the output multiplied by -1 when hoping to maximize.

```
import utils
import optimizer


if __name__ == '__main__':
    requirements, experimental_settings = utils.parse_requirements()
    hp_utils = utils.HyperparameterUtilities("sphere", experimental_settings=experimental_settings)
    opt = optimizer.TPE(hp_utils, **requirements)
    best_conf, best_performance = opt.optimize()
```


Run from termianl by (one example):

```
python main.py -dim 2 -par 1 -ini 3 -exp 0 -eva 100 -res 0 -seed 0 -veb 1 -fre 1
```

where all the arguments are integer.

### dim (optional: Default is None)
The dimension of input space.
Only for benchmark functions.

### par (optional: Default is 1)
The number of parallel computer resources such as GPU or CPU.

### ini (Required)
The number of initial samplings.

### exp (optional: Default is 0)
The index of experiments.
Used only for indexing of experiments.

### eva (optional: Default is 100)
The maximum number of evaluations in an experiment.
If eva = 100, 100 configurations will be evaluated.

### res (optional: Default is 0(=False))
Whether restarting an experiment based on the previous experiment.
If 0, will remove the previous log files after you choose "y" at the caution.

### seed  (optional: Default is None)
The number to specify the random seed.

### veb (optional: Default is 1)
Whether print the result or not. If 0, not print.

### fre (optional: Default is 1)
Every print_freq iteration, the result will be printed.

### dat (supervised learning)
The name of dataset.

### cls (supervised learning)
The number of classes on a given task.

### img (optional: Default is None)
The pixel size of training data.

### sub (optional: Default is None)
How many percentages of training data to use in training (Must be between 0. and 1.).

## Optimizer
You can add whatever optimizers you would like to use in this basement.
By inheriting the `BaseOptimizer` object, you can use basic function needed to start HPO.
A small example follows below:

```
from optimizer.base_optimizer import BaseOptimizer


class OptName(BaseOptimizer):
    def __init__(self,
                 hp_utils,  # hyperparameter utility object
                 n_parallels=1,  # the number of parallel computer resourses
                 n_init=10,  # the number of initial sampling
                 n_experiments=0,  # the index of experiments. Used only to specify the path of log files.
                 max_evals=100,  # the number of maximum evaluations in an experiment
                 restart=True,  # Whether restarting the previous experiment or not. If False, removes the previous log files.
                 seed=None,  # The number to specify the seed for a random number generator.
                 verbose=True,  # Whether print the result or not.
                 print_freq=1  # Every print_freq iteration, the result will be printed.
                 **kwargs
                 ):

        # inheritance
        super().__init__(hp_utils,
                         n_parallels=n_parallels,
                         n_init=n_init,
                         n_experiments=n_experiments,
                         max_evals=max_evals,
                         restart=restart,
                         seed=seed,
                         verbose=verbose,
                         print_freq=print_freq
                         )

        # optimizer in BaseOptimizer object
        self.opt = self.sample

    def sample(self):
        """
        some procedures and finally returns a hyperparameter configuration
        this hyperparameter configuration must be on usual scales.
        """

        return hp_conf
```

## Hyperparameters of Objective Functions
Describe the details of hyperparameters in `params.json`.

### 1. First key (The name of an objective function)
The name of objective function and it corresponds to the name of objective function callable.

### 2. y_names
The names of the measurements of hyperparameter configurations

### 3. in_fmt
The format of input for the objective function. Either 'list' or 'dict'.

### 4. config
The information related to the hyperparameters.

#### 4-1. the name of each hyperparameter
Used when recording the hyperparameter configurations.

#### 4-2. lower, upper
The lower and upper bound of the hyperparameter.
Required only for float and integer parameters.

#### 4-3. dist (required anytime)
The distribution of the hyperparameter.
Either 'u' (uniform) or 'c' (categorical).

#### 4-4. q
The quantization parameter of a hyperparameter.
If omited, q is going to be None.
Either any float or integer value or 'None'.

#### 4-5. log
If searching on a log-scale space or not.
If 'True', on a log scale.
If omited or 'False', on a linear scale.

#### 4-6. var_type (required anytime)
The type of a hyperparameter.
Either 'int' or 'float' or 'str' or 'bool'.

#### 4-7. choices (required only if dist is 'c' (categorical) )
The choices of categorical parameters.
Have to be given by a list.

#### 4-8. ignore (optional: "True" or "False")
Whether ignoring the hyperparameter or not.

An example follows below.

```
{
    "sphere": {
      "y_names": ["loss"],
      "in_fmt": "list",
      "config": {
            "x": {
                "lower": -5.0, "upper": 5.0,
                "dist": "u", "var_type": "float"
            }
        }
    },
    "cnn": {
      "y_names": ["error", "cross_entropy"],
      "in_fmt": "dict",
      "config": {
            "batch_size": {
                "lower": 32, "upper": 256,
                "dist": "u", "log": "True",
                "var_type": "int"
            },
            "lr": {
                "lower": 5.0e-3, "upper": 5.0e-1,
                "dist": "u", "log": "True",
                "var_type": "float"
            },
            "momentum": {
                "lower": 0.8, "upper": 1.0,
                "dist": "u", "q": 0.1,
                "log": "False", "var_type": "float"
            },
            "nesterov": {
                "dist": "c", "choices": [True, False],
                "var_type": "bool", "ignore": "True"
            }
        }
    }
}
```

## Objective Functions
The target objective function in an experiment.
This function must receive the `gpu_id`, `hp_conf`, `save_path`, and `experimental_settings` from `BaseOptimizer` object and return the performance by a dictionary format.
An example of (`obj_functions/benchmarks/sphere.py`) follows below.


```
import numpy as np

"""
Parameters
----------
experimental_settings: dict
    The dict of experimental settings.

Returns
-------
_imp: callable
"""

def f(experimental_settings):
    def _imp():
        """
        Parameters
        ----------
        hp_conf: 1d list of hyperparameter value
            [the index for a hyperparameter]
        gpu_id: int
            the index of a visible GPU
        save_path: str
            The path to record training.

        Returns
        -------
        ys: dict
            keys are the name of performance measurements.
            values are the corresponding performance.
        """

        return {"loss": (np.array(hp_conf) ** 2).sum()}

    return _imp
```

Also, the keys and corresponding values of `experimental_settings` are as follows:

### dim: int
The dimension of input space.
Only for benchmark functions.

### dataset_name: str
The name of dataset.

### n_cls: int
The number of classes on a given task.

### image_size: int
The pixel size of training data.

### data_frac: float
How many percentages of training data to use in training (Must be between 0. and 1.).

### biased_cls: list of float
The size of this list must be the same as n_cls.
The i-th element of this list is the percentages of training data to use in learning.
