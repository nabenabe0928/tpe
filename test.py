import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from argparse import ArgumentParser as ArgPar
from main import start_opt
from optimize import sample_target, create_hyperparameter, save_evaluation, print_iterations
from objective_functions.train import func


def objective_func(model, num, n_cuda, n_jobs, n_startup_trials = 10):
    hp_dict = {}
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
    var_name = "ch3"
    hp_dict[var_name] = sample(create_hyperparameter("int", name = var_name, lower = 16, upper = 128, default_value = 64, log = True))
    var_name = "ch4"
    hp_dict[var_name] = sample(create_hyperparameter("int", name = var_name, lower = 16, upper = 128, default_value = 64, log = True))
    var_name = "drop_rate"
    hp_dict[var_name] = sample(CSH.UniformFloatHyperparameter(name = "float", lower = 0., upper = 1., default_value = 0.5, log = False))
    
    loss, acc = func(hp_dict, model, num, n_cuda, n_jobs)
    print_iterations(n_jobs, loss, acc)
    hp_dict["loss"] = loss
    save_evaluation(hp_dict, model, num, n_jobs)

    
if __name__ == "__main__":
    argp = ArgPar()
    argp.add_argument("-num", type = int, default = None)
    argp.add_argument("-parallel", type = int, default = None)
    argp.add_argument("-jobs", type = int, default = None)
    argp.add_argument("-re", type = int, default = None, choices = [0, 1])
    args = argp.parse_args()
        
    num = args.num
    n_parallels = args.parallel
    n_jobs = args.jobs
    rerun = bool(args.re)
    
    start_opt(obj = objective_func, model = "CNN", num = num, n_parallels = n_parallels, n_jobs = n_jobs, rerun = rerun)
