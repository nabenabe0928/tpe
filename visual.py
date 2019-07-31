import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import csv
import time
import ConfigSpace as CS
import ConfigSpace as CSH
from argparse import ArgumentParser as ArgPar
from multiprocessing import Lock
from sampler.tpe_sampler import TPESampler, distribution_type, transform_vals, get_evaluations
from sampler.parzen_estimator import ParzenEstimator, ParzenEstimatorParameters
from optimize import create_hyperparameter


def plot_EI(model, num, x_name, x_min, x_max, y_name = "loss", y_min = 0., y_max = 5., is_log = False, q = None):

    lock = Lock()
    cs = CS.ConfigurationSpace()
    var = CSH.UniformFloatHyperparameter(x_name, lower = x_min, upper = x_max, log = is_log)
    cs.add_hyperparameter(var)
    xs, ys = get_evaluations(model, num, x_name, lock, cs)
    tpe = TPESampler(model, num, var, 0, lock)
    lower_bound, upper_bound, is_log, q = tpe.hp.lower, tpe.hp.upper, tpe.hp.log, tpe.hp.q
    _dist = distribution_type(tpe.target_cs, tpe.var_name)
    n_start = 10

    for n in range(n_start, len(ys)):
        tpe.hyperparameter = np.array(xs[:n])
        tpe.losses = np.array(ys[:n])
        lower_vals, upper_vals = tpe._split_observation_pairs()

        if n == n_start:
            lower_bound, upper_bound, lower_vals, upper_vals = transform_vals(lower_bound, upper_bound, lower_vals, upper_vals, _dist, q = q, is_log = is_log)
        else:
            _, __, lower_vals, upper_vals = transform_vals(1, 1, lower_vals, upper_vals, _dist, q = q, is_log = is_log)

        if is_log:
            samples = np.e ** (np.linspace(lower_bound, upper_bound, 100))
        else:
            samples = np.linspace(lower_bound, upper_bound, 100)

        parzen_estimator_lower = ParzenEstimator(lower_vals, lower_bound, upper_bound, tpe.parzen_estimator_parameters)
        log_lower = tpe._gmm_log_pdf(samples, parzen_estimator_lower, lower_bound, upper_bound, var_type = _dist, is_log=is_log, q = q)

        parzen_estimator_upper = ParzenEstimator(upper_vals, lower_bound, upper_bound, tpe.parzen_estimator_parameters)
        log_upper = tpe._gmm_log_pdf(samples, parzen_estimator_upper, lower_bound, upper_bound, var_type = _dist, is_log = is_log, q = q)

        samples, log_lower, log_upper = map(np.asarray, (samples, log_lower, log_upper))
        ei = log_lower - log_upper

        x_grid = np.linspace(lower_bound, upper_bound, 100)
        plt.figure()
        plt.xlim(lower_bound, upper_bound)
        plt.plot(x_grid, ei)
        plt.show()

def opt_movie(xs, fs, dt = 0.1, y_min = 0., y_max = 5.):
    n = len(fs)
    y_diff = 0.05 * (y_max - y_min)
    plt.ylim(y_min - y_diff, y_max + y_diff)
    plt.xlim(-0.05, 1.05)
    cnt = 0

    for t in range(n):
        plt.scatter(xs[t], fs[t], color = "blue")

        if t == 16 * cnt ** 2:
            plt.scatter(cnt * 0.04 , y_min - y_diff, color = "red")
            cnt += 1
        plt.pause(dt)

def get_evaluation(model, num, x_name, y_name, y_min = 0., y_max = 5., is_log = False):

    with open("evaluation/{}/{:0>3}/{}.csv".format(model, num, x_name)) as f:
        reader = list(csv.reader(f, delimiter = ","))

        if is_log:
            xs = np.array([np.log(float(row[1])) for row in reader])
        else:
            xs = np.array([float(row[1]) for row in reader])

        x_max = xs.max()
        x_min = xs.min()
        xs = (xs - x_min) / (x_max - x_min)

        idx = [int(row[0]) for row in reader]

    with open("evaluation/{}/{:0>3}/{}.csv".format(model, num, y_name)) as f:
        reader = csv.reader(f, delimiter = ",")
        ys = np.array([max(min(float(row[1]), y_max), y_min) for row in reader])

    order = np.argsort(idx)
    #gamma = gamma_function(len(order))
    #order_by_value = np.argsort(ys)
    #y_star = ys[order_by_value][gamma]
    xs = xs[order]
    ys = ys[order]

    return xs, ys #, y_star

if __name__ == "__main__":
    argp = ArgPar()
    argp.add_argument("-model", choices = os.listdir("evaluation"), required = True)
    argp.add_argument("-num", type = int, required = True)
    argp.add_argument("-xname", required = True)
    argp.add_argument("-xmax", type = float, required = True)
    argp.add_argument("-xmin", type = float, required = True)
    argp.add_argument("-yname", default = "loss")
    argp.add_argument("-ymax", type = float, default = 5.)
    argp.add_argument("-ymin", type = float, default = 0.)
    argp.add_argument("-log", type = int, default = 0, choices = [0, 1], required = True)
    args = argp.parse_args()

    model = args.model
    num = args.num
    xname = args.xname
    xmax = args.xmax
    xmin = args.xmin
    yname = args.yname
    ymax = args.ymax
    ymin = args.ymin
    is_log = bool(args.log)

    #xs, ys = get_evaluation(model, num, xname, yname, y_min = ymin, y_max = ymax, is_log = is_log)
    #opt_movie(xs, ys, y_min = ymin, y_max = ymax)
    plot_EI(model, num, x_name = xname, x_min = xmin, x_max = xmax, y_name = yname, y_min = ymin, y_max = ymax, is_log = is_log)