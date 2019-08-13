import numpy as np


def print_info(func, bounds, minimum):
    print("this is {} function.".format(func))
    print("boundary is {}".format(bounds))
    print("minimum is {}".format(minimum))


class ackley():
    def __init__(self, verbose=False):
        self.bounds = np.array([-32.768, 32.768])
        if verbose:
            print_info("ackley", self.bounds, 0)

    def f(self, x):
        t1 = 20
        t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))
        t3 = np.e
        t4 = - np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))
        return t1 + t2 + t3 + t4


class sphere():
    def __init__(self, verbose=False):
        self.bounds = np.array([-1, 1])
        if verbose:
            print_info("sphere", self.bounds, 0)

    def f(self, x):
        return np.sum(x ** 2)


class rosenbrock():
    def __init__(self, verbose=False):
        self.bounds = np.array([-5, 5])
        if verbose:
            print_info("rosenbrock", self.bounds, 0)

    def f(self, x):
        val = 0
        for i in range(0, len(x) - 1):
            t1 = 100 * (x[i + 1] - x[i] ** 2) ** 2
            t2 = (x[i] - 1) ** 2
            val += t1 + t2
        return val


class styblinski():
    def __init__(self, verbose=False):
        self.bounds = np.array([-5, 4])
        if verbose:
            print_info("styblinski-tang", self.bounds, -39.166165)

    def f(self, x):
        t1 = np.sum(x ** 4)
        t2 = - 16 * np.sum(x ** 2)
        t3 = 5 * np.sum(x)
        return 0.5 * (t1 + t2 + t3)


class k_tablet():
    def __init__(self, verbose=False):
        print("this is k-tablet function.")
        self.bounds = np.array([-5.12, 5.12])
        if verbose:
            print_info("k-tablet", self.bounds, 0)

    def f(self, x):
        k = int(np.ceil(len(x) / 4.0))
        t1 = np.sum(x[:k])
        t2 = 100 ** 2 * np.sum(x[k:] ** 2)
        return t1 + t2


class weighted_sphere():
    def __init__(self, verbose=False):
        self.bounds = np.array([-5.12, 5.12])
        if verbose:
            print_info("weighted sphere", self.bounds, 0)

    def f(self, x):
        f = np.array([(i + 1) * xi ** 2 for i, xi in enumerate(x)])
        return np.sum(f)


class different_power():
    def __init__(self, verbose=False):
        self.bounds = np.array([-1, 1])
        if verbose:
            print_info("different power", self.bounds, 0)

    def f(self, x):
        val = 0
        for i, v in enumerate(x):
            val += np.abs(v) ** (i + 2)
        return val


class griewank():
    def __init__(self, verbose=False):
        self.bounds = np.array([-600, 600])
        if verbose:
            print_info("griewank", self.bounds, 0)

    def f(self, x):
        w = np.array([1.0 / np.sqrt(i + 1) for i in range(len(x))])
        t1 = 1
        t2 = 1.0 / 4000.0 * np.sum(x ** 2)
        t3 = - np.prod(np.cos(x * w))
        return t1 + t2 + t3


class perm():
    def __init__(self, verbose=False):
        self.bounds = np.array([-1, 1])
        if verbose:
            print_info("perm", self.bounds, 0)

    def f(self, x):
        val_f = 0
        for j in range(len(x)):
            val = 0

            for i in range(len(x)):
                val += (i + 2) * (x[i] ** (j + 1) - ((1 / (i + 1)) ** (j + 1)))
            val_f += val ** 2

        return val_f


class rastrigin():
    def __init__(self, verbose=False):
        self.bounds = np.array([-5.12, 5.12])
        if verbose:
            print_info("rastringin", self.bounds, 0)

    def f(self, x):
        t1 = 10 * len(x)
        t2 = np.sum(x ** 2)
        t3 = - 10 * np.sum(np.cos(2 * np.pi * x))
        return t1 + t2 + t3


class schwefel():
    def __init__(self, verbose=False):
        self.bounds = np.array([-500, 500])
        if verbose:
            print_info("schwefel", self.bounds, -418.9829)

    def f(self, x):
        return - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class xin_she():
    def __init__(self, verbose=False):
        self.bounds = np.array([-2 * np.pi, 2 * np.pi])
        if verbose:
            print_info("xin-she yang", self.bounds, 0)

    def f(self, x):
        t1 = np.sum(np.abs(x))
        e1 = - np.sum(np.sin(x ** 2))
        t2 = np.exp(e1)
        return t1 * t2


class zakharov():
    def __init__(self, verbose=False):
        self.bounds = np.array([-100, 100])
        if verbose:
            print_info("zakharov", self.bounds, 0)

    def f(self, x):
        t1 = np.sum(x)
        w = np.array([i + 1 for i in range(len(x))])
        wx = np.dot(w, x)
        t2 = 0.5 ** 2 * wx ** 2
        t3 = 0.5 ** 4 * wx ** 4
        return t1 + t2 + t3


class rotated_hyper_ellipsoid():
    def __init__(self, verbose=False):
        self.bounds = np.array([-65.536, 65.536])
        if verbose:
            print_info("rotated hyper ellipsoid", self.bounds, 0)

    def f(self, x):
        s = 0
        for i in range(len(x)):
            s += np.sum(x[:i + 1] ** 2)
        return s


"""
class trid():
    def __init__(self, n_dim = 10):
        print("this is rotated trid function.")
        print("boundary is {}".format(self.boundaries))
        print("minimum is {}".format(0))

    def f(self, x):
        n_dim = len(x)
        self.boundaries = np.array([- n_dim ** 2, n_dim ** 2])

        t1 = np.sum( (x - 1) ** 2 )
        t2 = - np.sum( x[1:n_dim] * x[0:n_dim - 1] )

        return t1 + t2
"""


class dixon_price():
    def __init__(self, verbose=False):
        self.bounds = np.array([- 10, 10])
        if verbose:
            print_info("dixon price", self.bounds, 0)

    def f(self, x):
        n_dim = len(x)
        c = np.array([i + 2 for i in range(n_dim - 1)])
        t1 = (x[0] - 1) ** 2
        t2 = np.sum(c * (2 * x[1:n_dim] ** 2 - x[0:n_dim - 1]) ** 2)

        return t1 + t2


class levy():
    def __init__(self, verbose=False):
        self.bounds = np.array([- 10, 10])
        if verbose:
            print_info("levy", self.bounds, 0)

    def f(self, x):
        w = np.array(1. + (x - 1) / 4.)
        t1 = np.sin(np.pi * w[0]) ** 2
        t2 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        t3 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))

        return t1 + t2 + t3
