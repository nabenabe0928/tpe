import time
import multiprocessing
import csv
import os
import ConfigSpace as CS
from sampler.tpe_sampler import TPESampler
from argparse import ArgumentParser as ArgPar
from objective_functions.train import func

def save_evaluation(hp_dict, model, num, n_jobs):
    for var_name, hp in hp_dict.items():
        with open("evaluation/{}/{:0>3}/{}.csv".format(model, num, var_name), "a", newline = "") as f:
            writer = csv.writer(f, delimiter = ",", quotechar = "'")
            writer.writerow([n_jobs, hp])
    
def objective_func(model, num, n_cuda, n_jobs, config_space, n_startup_trials = 10):
    ###### change here to adopt the conditional parameters
    if n_jobs < n_startup_trials:
        for _ in range(n_jobs + 1):
            hp_dict = config_space.sample_configuration().get_dictionary()
        
    else:
        config_dict = config_space._hyperparameters
        for var_name, hp_info in config_dict.items():
            target_cs = CS.ConfigurationSpace().add_hyperparameter(hp_info)
            hp_info = TPESampler(model, num, target_cs, n_jobs, n_startup_trials = n_startup_trials).sample()
    
    loss, acc = func(hp_dict, model, num, n_cuda, n_jobs)
    
    hp_dict["loss"] = loss

    save_evaluation(hp_dict, model, num, n_jobs)

    print("")
    print("###################")
    print("# evaluation{: >5} #".format(n_jobs))
    print("###################")
    print("loss: {:.4f} acc: {:.2f}%".format(loss, acc * 100))

def optimize(model, num, config_space, max_jobs = 100, n_parallels = None):
    if n_parallels == None or n_parallels <= 1:
        _optimize_sequential(model, num, config_space, max_jobs = max_jobs)
    else:
        _optimize_parallel(model, num, config_space, max_jobs = max_jobs, n_parallels = n_parallels)
    

def _optimize_sequential(model, num, config_space, max_jobs = 100):
    n_jobs = 0

    while True:
        n_cuda = 0

        objective_func(model, num, n_cuda, n_jobs, config_space)
        n_jobs += 1
            
        if n_jobs >= max_jobs:
            break

def _optimize_parallel(model, num, config_space, max_jobs = 100, n_parallels = 4):
    jobs = []
    n_runnings = 0
    n_jobs = 0

    while True:
        cudas = [False for _ in range(n_parallels)]
        if len(jobs) > 0:
            n_runnings = 0
            new_jobs = []
            for job in jobs:
                if job[1].is_alive():
                    new_jobs.append(job)
                    cudas[job[0]] = True
            jobs = new_jobs
            n_runnings = len(jobs)
        else:
            n_runnings = 0

        for _ in range(max(0, n_parallels - n_runnings)):
            n_cuda = cudas.index(False)
            p = multiprocessing.Process(target = objective_func, args = (model, num, n_cuda, n_jobs, config_space))
            p.start()
            jobs.append([n_cuda, p])
            n_jobs += 1
            
            if n_jobs >= max_jobs:
                break
            
            time.sleep(1.0e-3)
    
        if n_jobs >= max_jobs:
            break