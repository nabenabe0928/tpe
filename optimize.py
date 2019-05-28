import time
import multiprocessing
from sampler.tpe_sampler import TPESampler
from argparse import ArgumentParser as ArgPar
from objective_functions.train import func

def objective_func(model, num, n_cuda, n_jobs, config_space):
    hp_dict = TPESampler(model, num, config_space)
    func(hp_dict, model, num, n_cuda, n_jobs)

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