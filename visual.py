import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import ConfigSpace as CS
import ConfigSpace as CSH
from argparse import ArgumentParser as ArgPar
from multiprocessing import Lock
from sampler.tpe_sampler import TPESampler, distribution_type, transform_vals, get_evaluations
from sampler.parzen_estimator import ParzenEstimator


def normal_pdf(x, w, mu, sgm):
    return w / np.sqrt(2 * np.pi * sgm ** 2) * np.e ** (- 0.5 * ((x - mu) / sgm) ** 2)


def plot_EI_1d(model, num, x_name, x_min, x_max, y_name="loss", y_min=0., y_max=5., is_log=False, n_start=10, step=10):
    lock = Lock()
    cs = CS.ConfigurationSpace()
    var = CSH.UniformFloatHyperparameter(x_name, lower=x_min, upper=x_max, log=is_log)
    cs.add_hyperparameter(var)
    xs, ys = get_evaluations(model, num, x_name, lock, cs)
    tpe = TPESampler(model, num, var, 0, lock)
    lb, ub = x_min, x_max
    _dist = distribution_type(tpe.target_cs, tpe.var_name)
    n_grids = 100
    lb, ub, _, __ = transform_vals(lb, ub, np.array([]), np.array([]), _dist, is_log=is_log)
    x_grid = np.linspace(lb, ub, n_grids)

    for n in range(n_start, len(ys), step):
        tpe.hyperparameter = np.array(xs[:n])
        tpe.losses = np.array(ys[:n])
        lvs, uvs = tpe._split_observation_pairs()
        _, __, lvs, uvs = transform_vals(1, 1, lvs, uvs, _dist, is_log=is_log)
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
        plot_and_get_ei(x_grid, samples, lvs, uvs, lb, ub, tpe, _dist, is_log, do_plot=True)


def plot_EI_2d(model, num, xnames, lbs, ubs, is_logs, y_name="loss", y_min=0., y_max=5., n_start=10, step=10):
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
    lbs[0], ubs[0], _, __ = transform_vals(lbs[0], ubs[0], np.array([]), np.array([]), _dists[0], is_log=is_logs[0])
    lbs[1], ubs[1], _, __ = transform_vals(lbs[1], ubs[1], np.array([]), np.array([]), _dists[1], is_log=is_logs[1])
    n_grids = 100
    grids = [np.linspace(lb, ub, n_grids) for lb, ub in zip(lbs, ubs)]
    # x_grid, y_grid = np.meshgrid(np.linspace(lbs[0], ubs[0], n_grids), np.linspace(lbs[1], ubs[1], n_grids))

    for n in range(n_start, len(ys), step):
        eis = []
        print("### {}".format(n))
        plt.figure()
        for tpe, xs, _dist, is_log, grid, lb, ub in zip(tpes, xns, _dists, is_logs, grids, lbs, ubs):
            tpe.hyperparameter = np.array(xs[:n])
            tpe.losses = np.array(ys[:n])
            lvs, uvs = tpe._split_observation_pairs()
            _, __, lvs, uvs = transform_vals(1, 1, lvs, uvs, _dist, is_log=is_log)
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


def plot_and_get_ei(grid, samples, lvs, uvs, lb, ub, tpe, _dist, is_log, do_plot=False):
    pe_lower = ParzenEstimator(lvs, lb, ub, tpe.parzen_estimator_parameters)
    log_lower = tpe._gmm_log_pdf(samples, pe_lower, lb, ub, var_type=_dist, is_log=is_log)
    pe_upper = ParzenEstimator(uvs, lb, ub, tpe.parzen_estimator_parameters)
    log_upper = tpe._gmm_log_pdf(samples, pe_upper, lb, ub, var_type=_dist, is_log=is_log)
    ei = np.array(log_lower) - np.array(log_upper)

    if do_plot:
        plot_pdf(pe_lower, grid, lb, ub, do_plot=True, axv=True, c="red")
        plot_pdf(pe_upper, grid, lb, ub, do_plot=False, axv=False, c="blue")
        plt.ylim(-2, 2)
        plt.plot(grid, ei, color="green")
        plt.show()

    return ei


def plot_pdf(PE, grid, lb, ub, do_plot=False, axv=False, c="red"):
    ws = PE.weights
    ms = PE.mus
    ss = PE.sigmas
    p = np.sum(ws * (TPESampler._normal_cdf(ub, ms, ss) - TPESampler._normal_cdf(lb, ms, ss)))
    SUM = np.zeros(len(grid))

    for w, m, s in zip(ws, ms, ss):
        base = normal_pdf(grid, w, m, s) / p
        SUM += base
        if do_plot:
            plt.plot(grid, base, color=c, linestyle="--")
    plt.plot(grid, (- 0.5 * lb + 0.5 * ub) * SUM, color=c)
    if axv:
        for m in ms:
            plt.axvline(x=m, color=c, linestyle="--")


def get_var_infos(model, var_name=None):
    with open("hps_dict/{}.csv".format(model), newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            if row["name"] == var_name:
                m = float(row["lower"])
                M = float(row["upper"])
                log = eval(row["log"])
            if var_name is None:
                m = float(row["lower"])
                M = float(row["upper"])
                log = eval(row["log"])

    return m, M, log


if __name__ == "__main__":
    argp = ArgPar()
    argp.add_argument("-model", choices=os.listdir("evaluation"), required=True)
    argp.add_argument("-num", type=int, required=True)
    argp.add_argument("-x1name", required=True)
    argp.add_argument("-x2name", default=None)
    argp.add_argument("-yname", default="loss")
    argp.add_argument("-ymax", type=float, default=5.)
    argp.add_argument("-ymin", type=float, default=0.)
    argp.add_argument("-start", type=int, default=10)
    argp.add_argument("-step", type=int, default=10)
    argp.add_argument("-mode", type=int, default=1, choices=[1, 2])

    args = argp.parse_args()

    model = args.model
    num = args.num
    x1name = args.x1name
    x2name = args.x2name

    if x1name[0] == "x" and x2name[0] == "x":
        x1min, x1max, is_log1 = get_var_infos(model)
        print(x1min, x1max, is_log1)
        x2min, x2max, is_log2 = x1min, x1max, is_log1
    else:
        x1min, x1max, is_log1 = get_var_infos(model, x1name)
        x2min, x2max, is_log2 = get_var_infos(model, x2name)

    yname = args.yname
    ymax = args.ymax
    ymin = args.ymin
    n_start = args.start
    step = args.step
    mode = args.mode

    if mode == 1:
        plot_EI_1d(model, num, x_name=x1name, x_min=x1min, x_max=x1max, y_name=yname, y_min=ymin, y_max=ymax, is_log=is_log1, n_start=n_start, step=step)
    elif mode == 2:
        plot_EI_2d(model, num, xnames=[x1name, x2name], lbs=[x1min, x2min], ubs=[x1max, x2max], is_logs=[is_log1, is_log2],
                   y_name=yname, y_min=ymin, y_max=ymax, n_start=n_start, step=step)
