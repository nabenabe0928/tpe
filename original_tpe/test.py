from argparse import ArgumentParser as ArgPar
from main import start_opt
from optimize import sample_target, create_hyperparameter, save_evaluation, print_iterations
#from objective_functions.square_func import func
from objective_functions.train import func

def objective_func(model, num, n_cuda, n_jobs, n_startup_trials = 10):
    hp_dict = {}
    sample = sample_target(model, num, n_jobs)

    for i in range(20):
        var_name = "x{}".format(i)
        hp_dict[var_name] = sample(create_hyperparameter("float", name = var_name, lower = -10, upper = 10, default_value = 5.0, log = False))
    
    loss = func(hp_dict, model, num, n_cuda, n_jobs)
    print_iterations(n_jobs, loss)
    hp_dict["loss"] = loss
    save_evaluation(hp_dict, model, num, n_jobs)

    
if __name__ == "__main__":
    argp = ArgPar()
    argp.add_argument("-model", default = "square")
    argp.add_argument("-num", type = int, default = None)
    argp.add_argument("-parallel", type = int, default = None)
    argp.add_argument("-jobs", type = int, default = None)
    argp.add_argument("-re", type = int, default = None, choices = [0, 1])
    args = argp.parse_args()
        
    num = args.num
    n_parallels = args.parallel
    n_jobs = args.jobs
    rerun = bool(args.re)
    model = args.model
    
    start_opt(obj = objective_func, model = model, num = num, n_parallels = n_parallels, n_jobs = n_jobs, rerun = rerun)
