import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from argparse import ArgumentParser as ArgPar
from main import main

def search_config():
    config_space = CS.ConfigurationSpace()
    hps = []

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
                     
    hps.append(CSH.UniformIntegerHyperparameter(name = "batch_size", lower = 32, upper = 256, default_value = 128, log = True))
    hps.append(CSH.UniformFloatHyperparameter(name = "lr", lower = 5.0e-3, upper = 5.0e-1, default_value = 5.0e-2, log = True))
    hps.append(CSH.UniformFloatHyperparameter(name = "momentum", lower = 0.9, upper = 1.0, default_value = 0.9, log = False))
    hps.append(CSH.UniformFloatHyperparameter(name = "weight_decay", lower = 5.0e-6, upper = 5.0e-2, default_value = 5.0e-4, log = True))
    hps.append(CSH.UniformIntegerHyperparameter(name = "ch1", lower = 16, upper = 128, default_value = 32, log = True))
    hps.append(CSH.UniformIntegerHyperparameter(name = "ch2", lower = 16, upper = 128, default_value = 32, log = True))
    hps.append(CSH.UniformIntegerHyperparameter(name = "ch3", lower = 16, upper = 128, default_value = 64, log = True))
    hps.append(CSH.UniformIntegerHyperparameter(name = "ch4", lower = 16, upper = 128, default_value = 64, log = True))
    hps.append(CSH.UniformFloatHyperparameter(name = "drop_rate", lower = 0., upper = 1., default_value = 0.5, log = False))
    #hps.append(CSH.CategoricalHyperparameter(name = "dummy", choices = ["apple", "orange", "grape"], default_value = "apple"))
    
    config_space.add_hyperparameters(hps)

    main(config_space, model = "CNN", num = num, n_parallels = n_parallels, n_jobs = n_jobs, rerun = rerun)    
    
if __name__ == "__main__":
    search_config()