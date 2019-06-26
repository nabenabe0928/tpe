# TPE

## Requirements
・python3.7

・ConfigSpace

`pip install ConfigSpace`

## Required Argument when implementing

### model
the name used when saving the hyperparamters 

### num
the experiment number (You can save up to 1000 experiments)

### jobs
the number of evaluations in one optimization

### parallel
the maximum number of evaluation which can be run at the same time.

### re
restart the experiment or not (in case the experiment suddenly stops, but you have the log file)

The log files of each hyperparameters are stocked in `tpe/evaluations/[model]/[num]/[Hyperparameter].csv`

## How to define objective function

In the objective function, you have to define each parameters as follows:

```py
hp_dict = {}

# get the sample function and create the location where you save the hyperparameters
sample = sample_target(model, num, n_jobs)

var_name = "batch_size"
hp_dict[var_name] = sample(create_hyperparameter("int", name = var_name, lower = 32, upper = 256, default_value = 128, log = True))
var_name = "lr"
hp_dict[var_name] = sample(create_hyperparameter("float", name = var_name, lower = 5.0e-3, upper = 5.0e-1, default_value = 5.0e-2, log = True))
var_name = "momentum"
hp_dict[var_name] = sample(create_hyperparameter("float", name = var_name, lower = 0.9, upper = 1.0, default_value = 0.9, log = False))
var_name = "weight_decay"
hp_dict[var_name] = sample(create_hyperparameter("float", name = var_name, lower = 5.0e-6, upper = 5.0e-2, default_value = 5.0e-4, log = True))
var_name = "ch1"
hp_dict[var_name] = sample(create_hyperparameter("int", name = var_name, lower = 16, upper = 128, default_value = 32, log = True))
var_name = "ch2"
hp_dict[var_name] = sample(create_hyperparameter("int", name = var_name, lower = 16, upper = 128, default_value = 32, log = True))
var_name = "drop_rate"
hp_dict[var_name] = sample(create_hyperparameter("float", name = var_name, lower = 0., upper = 1., default_value = 0.5, log = False))
```
