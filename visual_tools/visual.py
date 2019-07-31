import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import csv
import time
from argparse import ArgumentParser as ArgPar

def gamma_function(n):
    return int(np.ceil(min(np.sqrt(n) / 4, 25)))

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
    argp.add_argument("-yname", default = "loss")
    argp.add_argument("-ymax", type = float, default = 5.)
    argp.add_argument("-ymin", type = float, default = 0.)
    argp.add_argument("-log", type = int, default = 0, choices = [0, 1], required = True)
    args = argp.parse_args()

    model = args.model
    num = args.num
    xname = args.xname
    yname = args.yname
    ymax = args.ymax
    ymin = args.ymin
    is_log = bool(args.log)

    xs, ys = get_evaluation(model, num, xname, yname, y_min = ymin, y_max = ymax, is_log = is_log)
    opt_movie(xs, ys, y_min = ymin, y_max = ymax)
