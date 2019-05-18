# AIST_TPE

## Requirements
・python3.7

## Flow
1. main.py
Setting the enviromental variables and starting the for loop.
At the end of one loop setting new log file and calling main.sh

2. main.sh
main.sh calls sampler.py, env.py, run.sh sequently.

3. sampler.py
sampler.py collect the information of each hyperparameters and the configurations evaluated up to now. 
sampler.py creates the Density Estimator refering to the information.
Then, sample the configuration and choose the best configuration.

4. env.py
env.py gives the parameter's configuration to the command line (= run.sh). 

5. run.sh
run.sh calls the objective function and memorizes the configuration and corresponding output value in csv.
