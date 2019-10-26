import utils
import optimizer


if __name__ == '__main__':
    func_name = "sphere"
    requirements, experimental_settings = utils.parse_requirements()
    hp_utils = utils.HyperparameterUtilities(func_name, experimental_settings=experimental_settings)
    opt = optimizer.TPE(hp_utils, **requirements)
    best_conf, best_performance = opt.optimize()
