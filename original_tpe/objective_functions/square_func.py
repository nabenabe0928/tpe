import random

def func(hp_dict, model, num, n_cuda, n_jobs):
    loss = 0
    for v in hp_dict.values():
        loss += v ** 2
    
    return loss