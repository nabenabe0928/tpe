# AIST_TPE

## Requirements
・python3.7

・ConfigSpace

`pip install ConfigSpace`

## Required Argument when implementing

・model
the name used when saving the hyperparamters 

・num
the experiment number (You can save up to 1000 experiments)

・jobs
the number of evaluations in one optimization

・parallel
the maximum number of evaluation which can be run at the same time.

・re
restart the experiment or not (in case the experiment suddenly stops, but you have the log file)

The log files of each hyperparameters are stocked in `tpe/evaluations/[model]/[num]/[Hyperparameter].csv`

