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

def normal_pdf(x, w, mu, sgm):
    return w / np.sqrt(2 * np.pi * sgm ** 2) * np.e ** (- 0.5 * ((x - mu) / sgm) ** 2)


def plot_EI_1d(model, num, x_name, x_min, x_max, y_name = "loss", y_min = 0., y_max = 5., is_log = False, n_start = 10, step = 10):

    lock = Lock()
    cs = CS.ConfigurationSpace()
    var = CSH.UniformFloatHyperparameter(x_name, lower = x_min, upper = x_max, log = is_log)
    cs.add_hyperparameter(var)
    xs, ys = get_evaluations(model, num, x_name, lock, cs)
    tpe = TPESampler(model, num, var, 0, lock)
    lb, ub = x_min, x_max
    _dist = distribution_type(tpe.target_cs, tpe.var_name)
    n_grids = 100
    lb, ub, _, __ = transform_vals(lb, ub, np.array([]), np.array([]), _dist, is_log = is_log)
    x_grid = np.linspace(lb, ub, n_grids) 
    
    for n in range(n_start, len(ys), step):
        tpe.hyperparameter = np.array(xs[:n])
        tpe.losses = np.array(ys[:n])
        lvs, uvs = tpe._split_observation_pairs()
        _, __, lvs, uvs = transform_vals(1, 1, lvs, uvs, _dist, is_log = is_log)
        print("### {}".format(n)) 
        
        plt.figure()
        plt.xlim(lb, ub)
        plt.grid()
        if is_log:
            plt.title("{}: {} vs {} with Log Scale. # of better: {}".format(n, x_name, y_name, len(lvs)))
            samples = np.e ** (np.linspace(lb, ub, n_grids))
        else:
            plt.title("{}: {} vs {} with Usual Scale. # of better: {}".format(n, x_name, y_name, len(lvs)))
            samples = np.linspace(lb, ub, n_grids)

        plot_and_get_ei(x_grid, samples, lvs, uvs, lb, ub, tpe, _dist, is_log, do_plot = True)


def plot_EI_2d(model, num, xnames, lbs, ubs, is_logs, y_name = "loss", y_min = 0., y_max = 5., n_start = 10, step = 10):
    lock = Lock()
    cs = CS.ConfigurationSpace()
    hps = [CSH.UniformFloatHyperparameter(xn, lower=lb, upper=ub, log=xl) for xn, lb, ub, xl in zip(xnames, lbs, ubs, is_logs)]
    cs.add_hyperparameters(hps)
    xns = []
    tpes = []
    _dists = []
    for xname, lb, ub, is_log, hp in zip(xnames, lbs, ubs, is_logs, hps):
        xs, ys = get_evaluations(model, num, xname, lock, cs)
        xns.append(xs)
        tpes.append(TPESampler(model, num, hp, 0, lock))
        _dists.append(distribution_type(tpes[-1].target_cs, tpes[-1].var_name))
    lbs[0], ubs[0], _, __ = transform_vals(lbs[0], ubs[0], np.array([]), np.array([]), _dists[0], is_log = is_logs[0])
    lbs[1], ubs[1], _, __ = transform_vals(lbs[1], ubs[1], np.array([]), np.array([]), _dists[1], is_log = is_logs[1]) 
    n_grids = 100
    grids = [np.linspace(lb, ub, n_grids) for lb, ub in zip(lbs, ubs)]
    #x_grid, y_grid = np.meshgrid(np.linspace(lbs[0], ubs[0], n_grids), np.linspace(lbs[1], ubs[1], n_grids))

    for n in range(n_start, len(ys), step):
        eis = []
        print("### {}".format(n))
        plt.figure()
        
        for tpe, xs, _dist, is_log, grid, lb, ub in zip(tpes, xns, _dists, is_logs, grids, lbs, ubs):
            tpe.hyperparameter = np.array(xs[:n])
            tpe.losses = np.array(ys[:n])
            lvs, uvs = tpe._split_observation_pairs()
            _, __, lvs, uvs = transform_vals(1, 1, lvs, uvs, _dist, is_log = is_log)
            
            if is_log:
                samples = np.e ** (np.linspace(lb, ub, n_grids))
            else:
                samples = np.linspace(lb, ub, n_grids)
            
            eis.append(np.array(plot_and_get_ei(grid, samples, lvs, uvs, lb, ub, tpe, _dist, is_log)))

        eis = np.array(eis)
        x, y = np.meshgrid(grids[0], grids[1])
        f = np.array([[eis[0][j] + eis[1][i] for j in range(n_grids)] for i in range(n_grids)])
        contour = plt.contourf(x, y, f, 100)
        plt.colorbar(contour)
        plt.show()

def plot_and_get_ei(grid, samples, lvs, uvs, lb, ub, tpe, _dist, is_log, do_plot = False):
    pe_lower = ParzenEstimator(lvs, lb, ub, tpe.parzen_estimator_parameters)
    log_lower = tpe._gmm_log_pdf(samples, pe_lower, lb, ub, var_type = _dist, is_log=is_log)
    pe_upper = ParzenEstimator(uvs, lb, ub, tpe.parzen_estimator_parameters)
    log_upper = tpe._gmm_log_pdf(samples, pe_upper, lb, ub, var_type = _dist, is_log = is_log)
    ei = np.array(log_lower) - np.array(log_upper)
    
    if do_plot:
        plot_pdf(pe_lower, grid, lb, ub, do_plot = True, axv = True, c = "red")
        plot_pdf(pe_upper, grid, lb, ub, do_plot = False, axv = False, c = "blue")
        plt.ylim(-2, 2)
        plt.plot(grid, ei, color = "green")
        plt.show()

    return ei
    
        
def plot_pdf(PE, grid, lb, ub, do_plot = False, axv = False, c = "red"):
    ws = PE.weights
    ms = PE.mus
    ss = PE.sigmas
    p = np.sum(ws * (TPESampler._normal_cdf(ub, ms, ss) - TPESampler._normal_cdf(lb, ms, ss)))
    SUM = np.zeros(len(grid))
    
    for w, m, s in zip(ws, ms, ss):
        base = normal_pdf(grid, w, m, s) / p
        SUM += base
        if do_plot:
            plt.plot(grid, base, color = c, linestyle = "--")
    plt.plot(grid, (- 0.5 * lb + 0.5 * ub) * SUM, color = c)
    
    if axv:
        for m in ms:
            plt.axvline(x = m, color = c, linestyle = "--")

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
    argp.add_argument("-x1name", required = True)
    argp.add_argument("-x1max", type = float, required = True)
    argp.add_argument("-x1min", type = float, required = True)
    argp.add_argument("-log1", type = int, default = 0, choices = [0, 1], required = True)
    argp.add_argument("-x2name", default = None)
    argp.add_argument("-x2max", type = float, default = None)
    argp.add_argument("-x2min", type = float, default = None)
    argp.add_argument("-log2", type = int, default = 0, choices = [0, 1])
    argp.add_argument("-yname", default = "loss")
    argp.add_argument("-ymax", type = float, default = 5.)
    argp.add_argument("-ymin", type = float, default = 0.)
    argp.add_argument("-start", type = int, default = 10)
    argp.add_argument("-step", type = int, default = 10)
    argp.add_argument("-mode", type = int, default = 1, choices = [1, 2])
    
    args = argp.parse_args()

    model = args.model
    num = args.num
    x1name = args.x1name
    x1max = args.x1max
    x1min = args.x1min
    is_log1 = bool(args.log1)
    x2name = args.x2name
    x2max = args.x2max
    x2min = args.x2min
    is_log2 = bool(args.log2)
    yname = args.yname
    ymax = args.ymax
    ymin = args.ymin
    n_start = args.start
    step = args.step
    mode = args.mode

    #xs, ys = get_evaluation(model, num, xname, yname, y_min = ymin, y_max = ymax, is_log = is_log)
    #opt_movie(xs, ys, y_min = ymin, y_max = ymax)

    if mode == 1:
        plot_EI_1d(model, num, x_name = x1name, x_min = x1min, x_max = x1max, y_name = yname, y_min = ymin, y_max = ymax, is_log = is_log1, n_start = n_start, step = step)
    elif mode == 2:
        plot_EI_2d(model, num, xnames = [x1name, x2name], lbs = [x1min, x2min], ubs = [x1max, x2max], is_logs = [is_log1, is_log2], y_name = yname, y_min = ymin, y_max = ymax, n_start = n_start, step = step)
