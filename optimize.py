import time
import multiprocessing
import csv
import os
from sampler.tpe_sampler import TPESampler
from argparse import ArgumentParser as ArgPar
from objective_functions.train import func

def save_evaluation(hp_dict, model, num):
    
    if os.path.isfile("evaluation/{}/evaluation{:0>3}.csv".format(model, num)):
    
        with open("evaluation/{}/evaluation{:0>3}.csv".format(model, num), "r", newline = "") as f:
            head = list(csv.reader(f, delimiter = ",", quotechar = "'"))[0]
        
        with open("evaluation/{}/evaluation{:0>3}.csv".format(model, num), "a", newline = "") as f:
            writer = csv.writer(f, delimiter = ",", quotechar = "'")
            row = [hp_dict[k] for k in head]
            writer.writerow(row)
    else:
        with open("evaluation/{}/evaluation{:0>3}.csv".format(model, num), "w", newline = "") as f:
            writer = csv.DictWriter(f, delimiter = ",", quotechar = "'", fieldnames = hp_dict.keys())
            writer.writeheader()
            writer.writerow(hp_dict)

def objective_func(model, num, n_cuda, n_jobs, config_space, n_startup_trials = 10):

    if n_jobs < n_startup_trials:
        for _ in range(n_jobs + 1):
            hp_dict = config_space.sample_configuration().get_dictionary()
        
    else:
        hp_dict = TPESampler(model, num, config_space, n_jobs, n_startup_trials = n_startup_trials).sample()
    loss, acc = func(hp_dict, model, num, n_cuda, n_jobs)
    
    hp_dict["loss"] = loss

    save_evaluation(hp_dict, model, num)

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